"""
Trainer for SMILES WAE model.

Training loop with combined reconstruction + MMD loss, mirroring the
structure of AutoencoderTrainer but using the WAE forward pass.
"""

import logging

import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from deepchemography.shared import Logger
from deepchemography.shared.losses import wae_mmd_gaussianprior
from deepchemography.smiles.trainer import CircularBuffer, set_torch_seed_to_all_gens

logger = logging.getLogger(__name__)


class SmilesWAETrainer:
    """
    Trainer for SMILES WAE model.

    Uses combined loss: recon_loss + lambda_mmd * mmd_loss

    Args:
        config: Dict with training and model hyperparameters.
                Expected keys: 'training', 'wae_mmd'.
    """

    def __init__(self, config):
        self.config = config
        self.train_cfg = config['training']
        self.mmd_cfg = config['wae_mmd']

    def get_collate_fn(self, model):
        """Get collate function for DataLoader."""
        def collate(data):
            data.sort(key=len, reverse=True)
            tensors = [model.string2tensor(string, device='cpu')
                       for string in data]
            return tensors

        return collate

    def get_dataloader(self, model, data, shuffle=True):
        """Create DataLoader for training or validation."""
        return DataLoader(
            data,
            batch_size=self.train_cfg['n_batch'],
            shuffle=shuffle,
            num_workers=0,
            collate_fn=self.get_collate_fn(model),
        )

    def _train_epoch(self, model, epoch, tqdm_data, optimizer=None):
        """
        Train or evaluate for one epoch.

        Returns:
            Dict with epoch metrics.
        """
        if optimizer is None:
            model.eval()
        else:
            model.train()

        recon_buf = CircularBuffer(100)
        mmd_buf = CircularBuffer(100)
        total_buf = CircularBuffer(100)

        sigma = self.mmd_cfg['sigma']
        kernel = self.mmd_cfg['kernel']
        lambda_mmd = self.mmd_cfg['lambda_mmd']

        for input_batch in tqdm_data:
            input_batch = tuple(data.to(model.device) for data in input_batch)

            # Forward pass
            recon_loss, mu, logvar, z = model(input_batch)

            # MMD loss
            mmd_loss = wae_mmd_gaussianprior(z, sigma=sigma, kernel=kernel)

            # Combined loss
            loss = recon_loss + lambda_mmd * mmd_loss

            if not torch.isfinite(loss):
                logger.warning(f"Non-finite loss: recon={recon_loss.item():.4f}, "
                               f"mmd={mmd_loss.item():.4f}, skipping batch")
                continue

            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()

                grad_norm = clip_grad_norm_(
                    model.parameters(),
                    self.train_cfg['clip_grad'],
                )

                if grad_norm > self.train_cfg['clip_grad'] * 10:
                    logger.warning(f"Large gradient norm: {grad_norm:.2f}")

                optimizer.step()

            recon_buf.add(recon_loss.item())
            mmd_buf.add(mmd_loss.item())
            total_buf.add(loss.item())

            lr = optimizer.param_groups[0]['lr'] if optimizer is not None else 0
            tqdm_data.set_postfix_str(
                f'loss={total_buf.mean():.5f} recon={recon_buf.mean():.5f} '
                f'mmd={mmd_buf.mean():.5f} lr={lr:.6f}'
            )

        lr = optimizer.param_groups[0]['lr'] if optimizer is not None else 0

        postfix = {
            'epoch': epoch,
            'lr': lr,
            'recon_loss': recon_buf.mean(),
            'mmd_loss': mmd_buf.mean(),
            'total_loss': total_buf.mean(),
            'mode': 'Eval' if optimizer is None else 'Train',
        }

        return postfix

    def _calculate_accuracy_subset(self, model, data_loader, max_samples=1000):
        """Calculate reconstruction accuracy on a subset of data."""
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for input_batch in data_loader:
                if max_samples > 0 and total >= max_samples:
                    break

                input_batch = tuple(data.to(model.device) for data in input_batch)

                for sample in input_batch:
                    if max_samples > 0 and total >= max_samples:
                        break

                    z = model.encode([sample])
                    reconstructed = model.sample(n_batch=1, z=z, temp=1.0)
                    original = model.tensor2string(sample)
                    if reconstructed[0] == original:
                        correct += 1
                    total += 1

        accuracy = 100 * correct / total if total > 0 else 0.0
        return accuracy

    def _train(self, model, train_loader, val_loader=None, log=None,
               model_save=None, log_file=None):
        """Main training loop."""
        device = model.device
        n_epochs = self.train_cfg['n_epochs']

        optimizer = optim.Adam(model.parameters(), lr=self.train_cfg['lr'])

        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=self.train_cfg['lr_factor'],
            patience=self.train_cfg['lr_patience'],
            min_lr=self.train_cfg['lr_min'],
        )

        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_model_state = None
        prev_lr = self.train_cfg['lr']

        model.zero_grad()
        for epoch in range(n_epochs):
            # Training
            tqdm_data = tqdm(train_loader, desc=f'Training (epoch #{epoch})')
            postfix = self._train_epoch(model, epoch, tqdm_data, optimizer)
            if log is not None:
                log.append(postfix)
                if log_file is not None:
                    log.save(log_file)

            # Validation
            if val_loader is not None:
                tqdm_data = tqdm(val_loader, desc=f'Validation (epoch #{epoch})')
                val_postfix = self._train_epoch(model, epoch, tqdm_data)
                val_loss = val_postfix['total_loss']

                # Accuracy every 5 epochs
                if epoch % 5 == 0:
                    logger.info("Calculating accuracy on 500 samples...")
                    val_accuracy = self._calculate_accuracy_subset(
                        model, val_loader, max_samples=500
                    )
                    val_postfix['accuracy'] = val_accuracy
                    logger.info(f"Validation accuracy: {val_accuracy:.2f}%")

                if log is not None:
                    log.append(val_postfix)
                    if log_file is not None:
                        log.save(log_file)

                scheduler.step(val_loss)

                current_lr = optimizer.param_groups[0]['lr']
                if current_lr < prev_lr:
                    logger.info(f"Learning rate reduced: {prev_lr:.6f} -> {current_lr:.6f}")
                    prev_lr = current_lr

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                    best_model_state = {k: v.cpu().clone()
                                        for k, v in model.state_dict().items()}
                    if model_save is not None:
                        torch.save(best_model_state,
                                   model_save[:-3] + '_best.pt')
                    logger.info(f"New best validation loss: {best_val_loss:.4f}")
                else:
                    epochs_no_improve += 1
                    logger.info(f"No improvement for {epochs_no_improve} epoch(s). "
                                f"Best loss: {best_val_loss:.4f}")

                if epochs_no_improve >= self.train_cfg['early_stop_patience']:
                    logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                    break

            # Save periodic checkpoint
            if model_save is not None and epoch % 5 == 0:
                checkpoint = {k: v.cpu().clone()
                              for k, v in model.state_dict().items()}
                torch.save(checkpoint, model_save[:-3] + f'_{epoch:03d}.pt')

        # Restore best model
        if best_model_state is not None:
            best_state_gpu = {k: v.to(device)
                              for k, v in best_model_state.items()}
            model.load_state_dict(best_state_gpu)
            logger.info(f"Restored best model with validation loss: {best_val_loss:.4f}")

    def fit(self, model, train_data, val_data=None,
            model_save=None, log_file=None):
        """
        Fit the SMILES WAE model on training data.

        Args:
            model: SmilesWAE instance
            train_data: List of SMILES strings
            val_data: Optional list of SMILES strings for validation
            model_save: Path to save model checkpoints (e.g. 'model.pt')
            log_file: Path to save training log

        Returns:
            Trained model
        """
        logger_obj = Logger() if log_file is not None else None

        train_loader = self.get_dataloader(model, train_data, shuffle=True)
        val_loader = (None if val_data is None
                      else self.get_dataloader(model, val_data, shuffle=False))

        self._train(model, train_loader, val_loader, logger_obj,
                    model_save=model_save, log_file=log_file)
        return model

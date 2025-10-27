"""
Autoencoder trainer for LSTM-based models.
"""
import logging
import torch
import torch.optim as optim
from tqdm.auto import tqdm
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from deepchemography.utils import OneHotVocab, Logger
from deepchemography.autoencoder.model import LSTMAutoencoder

logger = logging.getLogger(__name__)


class CircularBuffer:
    """Circular buffer for tracking recent values."""
    def __init__(self, size):
        self.size = size
        self.data = []
        self.idx = 0

    def add(self, value):
        if len(self.data) < self.size:
            self.data.append(value)
        else:
            self.data[self.idx] = value
            self.idx = (self.idx + 1) % self.size

    def mean(self):
        return sum(self.data) / len(self.data) if len(self.data) > 0 else 0.0


def set_torch_seed_to_all_gens(_):
    """Seed generator for DataLoader workers."""
    seed = torch.initial_seed() % (2**32 - 1)
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)


class AutoencoderTrainer:
    """Trainer for LSTM Autoencoder models."""
    
    def __init__(self, config):
        self.config = config
    
    @property
    def n_workers(self):
        """Number of DataLoader workers."""
        n_workers = self.config.n_workers
        return n_workers if n_workers != 1 else 0
    
    def get_collate_device(self, model):
        """Get device for collate function."""
        n_workers = self.n_workers
        return 'cpu' if n_workers > 0 else model.device
    
    def get_vocabulary(self, data):
        """
        Build vocabulary from training data.
        
        Args:
            data: list of SMILES strings
            
        Returns:
            OneHotVocab instance
        """
        logger.info("Building SMILES vocabulary...")
        vocab = OneHotVocab.from_data(data)
        logger.info(f"Vocabulary size: {len(vocab)}")
        return vocab
    
    def get_collate_fn(self, model):
        """Get collate function for DataLoader."""
        device = self.get_collate_device(model)
        
        def collate(data):
            data.sort(key=len, reverse=True)
            tensors = [model.string2tensor(string, device=device)
                      for string in data]
            return tensors
        
        return collate
    
    def get_dataloader(self, model, data, shuffle=True):
        """Create DataLoader for training or validation."""
        collate_fn = self.get_collate_fn(model)
        return DataLoader(
            data, 
            batch_size=self.config.n_batch,
            shuffle=shuffle,
            num_workers=self.n_workers, 
            collate_fn=collate_fn,
            worker_init_fn=set_torch_seed_to_all_gens
            if self.n_workers > 0 else None
        )
    
    def _train_epoch(self, model, epoch, tqdm_data, optimizer=None):
        """Train for one epoch."""
        if optimizer is None:
            model.eval()
        else:
            model.train()
        
        loss_values = CircularBuffer(self.config.n_last)
        
        for input_batch in tqdm_data:
            input_batch = tuple(data.to(model.device) for data in input_batch)
            
            # Forward pass
            loss = model(input_batch)
            
            # Check for NaN/Inf
            if not torch.isfinite(loss):
                logger.warning(f"Non-finite loss detected: {loss.item()}, skipping batch")
                continue
            
            # Backward pass
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                
                # Clip gradients
                grad_norm = clip_grad_norm_(
                    self.get_optim_params(model),
                    self.config.clip_grad
                )
                
                # Check for gradient explosion
                if grad_norm > self.config.clip_grad * 10:
                    logger.warning(f"Large gradient norm: {grad_norm:.2f}")
                
                optimizer.step()
            
            # Log metrics
            loss_values.add(loss.item())
            lr = (optimizer.param_groups[0]['lr']
                  if optimizer is not None
                  else 0)
            
            # Update progress bar
            loss_value = loss_values.mean()
            postfix = [f'loss={loss_value:.5f}', f'lr={lr:.5f}']
            tqdm_data.set_postfix_str(' '.join(postfix))
        
        postfix = {
            'epoch': epoch,
            'lr': lr,
            'recon_loss': loss_value,
            'mode': 'Eval' if optimizer is None else 'Train'
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
                        
                    z = model.forward_encoder([sample])
                    reconstructed = model.sample(n_batch=1, z=z, temp=1.0)
                    original = model.tensor2string(sample)
                    if reconstructed[0] == original:
                        correct += 1
                    total += 1
        
        accuracy = 100 * correct / total if total > 0 else 0.0
        return accuracy
    
    def get_optim_params(self, model):
        """Get trainable parameters."""
        return (p for p in model.autoencoder.parameters() if p.requires_grad)
    
    def _train(self, model, train_loader, val_loader=None, logger=None):
        """Main training loop."""
        device = model.device
        n_epoch = self.config.n_epochs
        
        # Setup optimizer
        optimizer = optim.Adam(
            self.get_optim_params(model),
            lr=self.config.lr_start
        )
        
        # Setup learning rate scheduler
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=self.config.lr_factor,
            patience=self.config.lr_patience,
            min_lr=self.config.lr_min
        )
        
        # Early stopping variables
        use_accuracy_for_stopping = self.config.early_stop_metric == 'accuracy'
        best_val_metric = 0.0 if use_accuracy_for_stopping else float('inf')
        epochs_no_improve = 0
        best_model_state = None
        prev_lr = self.config.lr_start
        
        model.zero_grad()
        for epoch in range(n_epoch):
            # Training epoch
            tqdm_data = tqdm(train_loader,
                            desc=f'Training (epoch #{epoch})')
            postfix = self._train_epoch(model, epoch, tqdm_data, optimizer)
            if logger is not None:
                logger.append(postfix)
                logger.save(self.config.log_file)
            
            # Validation epoch
            val_loss = None
            val_accuracy = None
            if val_loader is not None:
                tqdm_data = tqdm(val_loader,
                                desc=f'Validation (epoch #{epoch})')
                postfix = self._train_epoch(model, epoch, tqdm_data)
                val_loss = postfix['recon_loss']
                
                # Optionally calculate accuracy on subset
                if use_accuracy_for_stopping or epoch % 5 == 0:
                    logger.info(f"Calculating accuracy on {self.config.val_accuracy_samples} samples...")
                    val_accuracy = self._calculate_accuracy_subset(
                        model, val_loader, self.config.val_accuracy_samples
                    )
                    postfix['accuracy'] = val_accuracy
                    logger.info(f"Validation accuracy: {val_accuracy:.2f}%")
                
                if logger is not None:
                    logger.append(postfix)
                    logger.save(self.config.log_file)
                
                # Learning rate scheduler step
                scheduler.step(val_loss)
                
                # Check if learning rate was reduced
                current_lr = optimizer.param_groups[0]['lr']
                if current_lr < prev_lr:
                    logger.info(f"Learning rate reduced: {prev_lr:.6f} → {current_lr:.6f}")
                    prev_lr = current_lr
                
                # Check for improvement
                if use_accuracy_for_stopping:
                    current_metric = val_accuracy if val_accuracy is not None else 0.0
                    improved = current_metric > best_val_metric
                else:
                    current_metric = val_loss
                    improved = current_metric < best_val_metric
                
                if improved:
                    best_val_metric = current_metric
                    epochs_no_improve = 0
                    # Save best model
                    best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    if self.config.model_save is not None:
                        torch.save(best_model_state,
                                 self.config.model_save[:-3] + '_best.pt')
                    metric_name = "accuracy" if use_accuracy_for_stopping else "loss"
                    logger.info(f"✓ New best validation {metric_name}: {best_val_metric:.4f}")
                else:
                    epochs_no_improve += 1
                    metric_name = "accuracy" if use_accuracy_for_stopping else "loss"
                    logger.info(f"✗ No improvement for {epochs_no_improve} epoch(s). "
                               f"Best {metric_name}: {best_val_metric:.4f}")
                
                # Early stopping check
                if epochs_no_improve >= self.config.early_stop_patience:
                    logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                    metric_name = "accuracy" if use_accuracy_for_stopping else "loss"
                    logger.info(f"Best validation {metric_name}: {best_val_metric:.4f}")
                    break
            
            # Save periodic checkpoint
            if (self.config.model_save is not None) and \
                    (epoch % self.config.save_frequency == 0):
                checkpoint_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                torch.save(checkpoint_state,
                          self.config.model_save[:-3] + '_{0:03d}.pt'.format(epoch))
        
        # Restore best model
        if best_model_state is not None:
            best_model_state_gpu = {k: v.to(device) for k, v in best_model_state.items()}
            model.load_state_dict(best_model_state_gpu)
            metric_name = "accuracy" if use_accuracy_for_stopping else "loss"
            logger.info(f"Restored best model with validation {metric_name}: {best_val_metric:.4f}")
    
    def fit(self, model, train_data, val_data=None):
        """Fit the model on training data."""
        logger_obj = Logger() if self.config.log_file is not None else None
        
        train_loader = self.get_dataloader(model, train_data, shuffle=True)
        val_loader = None if val_data is None else self.get_dataloader(
            model, val_data, shuffle=False
        )
        
        self._train(model, train_loader, val_loader, logger_obj)
        return model


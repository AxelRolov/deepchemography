"""
SMILES Wasserstein Autoencoder (WAE) with GRU encoder/decoder.

Mirrors the PeptideWAE architecture adapted for character-level SMILES:
- Bidirectional GRU encoder with reparameterization (mu, logvar)
- GRU decoder with word dropout and teacher forcing
- MMD loss for latent regularization
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class WordDropout(nn.Module):
    """Word dropout layer that randomly replaces tokens with <unk>."""

    def __init__(self, p_word_dropout, unk_idx):
        super().__init__()
        self.p = p_word_dropout
        self.unk_idx = unk_idx

    def forward(self, x):
        """
        Apply word dropout: with prob p, set the token to unk_idx.

        Args:
            x: Token indices (mbsize x seq_len) or list of 1D tensors

        Returns:
            Token indices with random replacements
        """
        if not self.training or self.p == 0:
            return x

        data = x.clone().detach()
        mask = torch.from_numpy(
            np.random.binomial(1, p=self.p, size=tuple(data.size())).astype('uint8')
        ).to(x.device).bool()
        data[mask] = self.unk_idx
        return data


class SmilesWAE(nn.Module):
    """
    SMILES Wasserstein Autoencoder with GRU encoder/decoder.

    Uses character-level tokenization via OneHotVocab. Architecture mirrors
    PeptideWAE but adapted for variable-length SMILES strings.

    Args:
        vocab: OneHotVocab instance
        config: Dict with model hyperparameters (see wae_config.py)
    """

    def __init__(self, vocab, config):
        super().__init__()

        self.vocabulary = vocab
        self.config = config

        # Special token indices
        self.pad_idx = vocab.pad
        self.bos_idx = vocab.bos
        self.eos_idx = vocab.eos
        self.unk_idx = vocab.unk

        n_vocab = len(vocab)
        self.n_vocab = n_vocab

        # Dimensions from config
        self.z_dim = config['z_dim']
        self.emb_dim = config['emb_dim']
        self.max_seq_len = config['max_seq_len']

        enc_cfg = config['encoder']
        dec_cfg = config['decoder']

        h_dim = enc_cfg['h_dim']
        biGRU = enc_cfg['biGRU']
        enc_layers = enc_cfg['layers']
        enc_dropout = enc_cfg['p_dropout']
        biGRU_factor = 2 if biGRU else 1

        # Device tracking
        self.device = torch.device('cpu')

        # Word embeddings
        self.word_emb = nn.Embedding(n_vocab, self.emb_dim, self.pad_idx)

        # Encoder: bidirectional GRU
        self.encoder = nn.GRU(
            input_size=self.emb_dim,
            hidden_size=h_dim,
            num_layers=enc_layers,
            dropout=enc_dropout if enc_layers > 1 else 0,
            bidirectional=biGRU,
            batch_first=True,
        )
        self.biGRU = biGRU
        self.biGRU_factor = biGRU_factor
        self.enc_h_dim = h_dim

        # Reparameterization layers
        self.q_mu = nn.Linear(biGRU_factor * h_dim, self.z_dim)
        self.q_logvar = nn.Linear(biGRU_factor * h_dim, self.z_dim)

        # Decoder: GRU with z concatenated at each timestep
        self.decoder = nn.GRU(
            input_size=self.emb_dim + self.z_dim,
            hidden_size=self.z_dim,
            num_layers=1,
            batch_first=True,
        )

        # Word dropout for decoder inputs
        self.word_dropout = WordDropout(dec_cfg['p_word_dropout'], self.unk_idx)

        # Decoder output projection
        self.decoder_fc = nn.Sequential(
            nn.Dropout(dec_cfg['p_out_dropout']),
            nn.Linear(self.z_dim, n_vocab),
        )

        # Decoder initial hidden state from z
        self.decoder_h0 = nn.Linear(self.z_dim, self.z_dim)

    def to(self, device):
        """Override to() to update self.device when model is moved."""
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        return super().to(device)

    def forward_encoder(self, x):
        """
        Encode variable-length SMILES to latent (mu, logvar).

        Args:
            x: List of 1D LongTensors (variable-length token sequences)

        Returns:
            mu: (mbsize x z_dim)
            logvar: (mbsize x z_dim)
        """
        # Embed and pack
        x_emb = [self.word_emb(i_x) for i_x in x]
        x_packed = nn.utils.rnn.pack_sequence(x_emb, enforce_sorted=False)

        _, h = self.encoder(x_packed, None)

        if self.biGRU:
            # Concatenate forward and backward from top layer
            h = torch.cat((h[-2, :, :], h[-1, :, :]), dim=1)
        else:
            h = h[-1]

        mu = self.q_mu(h)
        logvar = self.q_logvar(h)
        return mu, logvar

    def sample_z(self, mu, logvar):
        """Reparameterization trick: z = mu + std * eps; eps ~ N(0, I)"""
        eps = torch.randn(mu.size(0), self.z_dim).to(self.device)
        return mu + torch.exp(logvar / 2) * eps

    def sample_z_prior(self, n_batch):
        """Sample z ~ p(z) = N(0, I)"""
        return torch.randn(n_batch, self.z_dim).to(self.device)

    def forward_decoder(self, x, z):
        """
        Decode with teacher forcing, computing reconstruction loss.

        Args:
            x: List of 1D LongTensors (variable-length token sequences)
            z: Latent vectors (mbsize x z_dim)

        Returns:
            recon_loss: Cross-entropy reconstruction loss (scalar)
        """
        lengths = [len(i_x) for i_x in x]

        # Pad sequences
        x_padded = nn.utils.rnn.pad_sequence(x, batch_first=True,
                                             padding_value=self.pad_idx)

        # Apply word dropout and embed
        x_dropped = self.word_dropout(x_padded)
        x_emb = self.word_emb(x_dropped)

        # Concatenate z at each timestep
        z_expanded = z.unsqueeze(1).repeat(1, x_emb.size(1), 1)
        x_input = torch.cat([x_emb, z_expanded], dim=-1)

        # Pack for efficient RNN processing
        x_input = nn.utils.rnn.pack_padded_sequence(
            x_input, lengths, batch_first=True, enforce_sorted=False
        )

        # Initialize decoder hidden state from z
        h_0 = self.decoder_h0(z).unsqueeze(0)  # (1 x mbsize x z_dim)

        # Decode
        output, _ = self.decoder(x_input, h_0)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

        # Project to vocab logits
        logits = self.decoder_fc(output)

        # Reconstruction loss: predict next token from current
        recon_loss = F.cross_entropy(
            logits[:, :-1].contiguous().view(-1, logits.size(-1)),
            x_padded[:, 1:].contiguous().view(-1),
            ignore_index=self.pad_idx,
        )

        return recon_loss

    def forward(self, x):
        """
        Full forward pass for training.

        Args:
            x: List of 1D LongTensors (variable-length token sequences)

        Returns:
            recon_loss: Reconstruction loss (scalar)
            mu: Encoder mean (mbsize x z_dim)
            logvar: Encoder log-variance (mbsize x z_dim)
            z: Sampled latent vectors (mbsize x z_dim)
        """
        mu, logvar = self.forward_encoder(x)
        z = self.sample_z(mu, logvar)
        recon_loss = self.forward_decoder(x, z)
        return recon_loss, mu, logvar, z

    def encode(self, x):
        """
        Encode to latent space (inference, returns mu).

        Args:
            x: List of 1D LongTensors

        Returns:
            z: Latent vectors (mbsize x z_dim), deterministic (mu)
        """
        with torch.no_grad():
            mu, _ = self.forward_encoder(x)
        return mu

    def sample(self, n_batch, max_len=None, z=None, temp=1.0, decode='sample'):
        """
        Generate SMILES from latent vectors.

        Args:
            n_batch: Number of sequences to generate
            max_len: Maximum generation length (default: config max_seq_len)
            z: Latent vectors (n_batch x z_dim), or None to sample from prior
            temp: Temperature for sampling
            decode: 'greedy' for argmax, 'sample' for stochastic

        Returns:
            List of SMILES strings
        """
        if max_len is None:
            max_len = self.max_seq_len

        with torch.no_grad():
            if z is None:
                z = self.sample_z_prior(n_batch)
            z = z.to(self.device)

            # Initialize decoder hidden state
            h = self.decoder_h0(z).unsqueeze(0)  # (1 x n_batch x z_dim)

            # Start with BOS token
            w = torch.full((n_batch,), self.bos_idx, dtype=torch.long,
                           device=self.device)
            result = torch.full((n_batch, max_len), self.pad_idx,
                                dtype=torch.long, device=self.device)
            result[:, 0] = self.bos_idx
            end_pads = torch.full((n_batch,), max_len, dtype=torch.long,
                                  device=self.device)
            eos_mask = torch.zeros(n_batch, dtype=torch.bool,
                                   device=self.device)

            for i in range(1, max_len):
                # Embed current token and concat z
                x_emb = self.word_emb(w).unsqueeze(1)  # (n_batch x 1 x emb_dim)
                z_step = z.unsqueeze(1)  # (n_batch x 1 x z_dim)
                x_input = torch.cat([x_emb, z_step], dim=-1)

                output, h = self.decoder(x_input, h)
                logits = self.decoder_fc(output.squeeze(1))

                if decode == 'greedy':
                    w = logits.argmax(dim=-1)
                else:
                    probs = F.softmax(logits / temp, dim=-1)
                    w = torch.multinomial(probs, 1)[:, 0]

                result[~eos_mask, i] = w[~eos_mask]
                i_eos_mask = ~eos_mask & (w == self.eos_idx)
                end_pads[i_eos_mask] = i + 1
                eos_mask = eos_mask | i_eos_mask

                if eos_mask.all():
                    break

        # Convert to strings
        sequences = []
        for i in range(n_batch):
            ids = result[i, :end_pads[i]].tolist()
            sequences.append(self.tensor2string(torch.tensor(ids)))

        return sequences

    def string2tensor(self, string, device='model'):
        """Convert SMILES string to tensor of token indices."""
        ids = self.vocabulary.string2ids(string, add_bos=True, add_eos=True)
        tensor = torch.tensor(
            ids, dtype=torch.long,
            device=self.device if device == 'model' else device,
        )
        return tensor

    def tensor2string(self, tensor):
        """Convert tensor of token indices to SMILES string."""
        ids = tensor.tolist()
        return self.vocabulary.ids2string(ids, rem_bos=True, rem_eos=True)

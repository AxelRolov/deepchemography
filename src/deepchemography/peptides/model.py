"""
Peptide Wasserstein Autoencoder (WAE) model.

Based on:
1. Hu, Zhiting, et al. "Toward controlled generation of text." ICML. 2017.
2. Bowman, Samuel R., et al. "Generating sentences from a continuous space." 2015.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from deepchemography.peptides.encoder import GRUEncoder
from deepchemography.peptides.decoder import GRUDecoder
from deepchemography.peptides.config import UNK_IDX, PAD_IDX, START_IDX, EOS_IDX
from deepchemography.peptides.utils import soft_embed, onehot_embed


class PeptideWAE(nn.Module):
    """
    Peptide Wasserstein Autoencoder with GRU encoder/decoder.

    Args:
        n_vocab: Vocabulary size
        max_seq_len: Maximum sequence length
        z_dim: Latent dimension
        c_dim: Condition dimension
        emb_dim: Embedding dimension
        encoder_config: Dict with encoder hyperparameters
        decoder_config: Dict with decoder hyperparameters
        pretrained_emb: Optional pretrained embeddings
        freeze_embeddings: Whether to freeze embeddings
    """

    def __init__(
        self,
        n_vocab: int,
        max_seq_len: int = 25,
        z_dim: int = 100,
        c_dim: int = 2,
        emb_dim: int = 150,
        encoder_config: dict = None,
        decoder_config: dict = None,
        pretrained_emb: torch.Tensor = None,
        freeze_embeddings: bool = False,
    ):
        super(PeptideWAE, self).__init__()

        self.MAX_SEQ_LEN = max_seq_len
        self.n_vocab = n_vocab
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.emb_dim = emb_dim

        # Device tracking
        self.device = torch.device('cpu')

        # Default configs
        if encoder_config is None:
            encoder_config = {
                'h_dim': 80,
                'biGRU': True,
                'layers': 1,
                'p_dropout': 0.0,
            }
        if decoder_config is None:
            decoder_config = {
                'p_word_dropout': 0.3,
                'p_out_dropout': 0.3,
                'skip_connections': False,
            }

        # Word embeddings
        self.word_emb = nn.Embedding(n_vocab, emb_dim, PAD_IDX)
        if pretrained_emb is not None:
            assert emb_dim == pretrained_emb.size(1), 'emb dim must match pretrained'
            self.word_emb.weight.data.copy_(pretrained_emb)
        if freeze_embeddings:
            self.word_emb.weight.requires_grad = False

        # Encoder
        self.encoder = GRUEncoder(
            emb_dim=emb_dim,
            z_dim=z_dim,
            **encoder_config,
        )

        # Decoder
        self.decoder = GRUDecoder(
            embedding=self.word_emb,
            emb_dim=emb_dim + z_dim + c_dim,
            output_dim=n_vocab,
            h_dim=z_dim + c_dim,
            **decoder_config,
        )

    def to(self, device):
        """Override to() to update self.device when model is moved."""
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        return super().to(device)

    def forward_encoder(self, inputs):
        """
        Encode inputs to latent space.

        Args:
            inputs: Batch of sequences (mbsize x seq_len) or
                    soft sequences (mbsize x seq_len x n_vocab)

        Returns:
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
        """
        if inputs.dim() == 2:
            inputs = self.word_emb(inputs)
        else:
            inputs = soft_embed(self.word_emb, inputs)
        return self.encoder(inputs)

    def sample_z(self, mu, logvar):
        """
        Reparameterization trick: z = mu + std * eps; eps ~ N(0, I)
        """
        eps = torch.randn(mu.size(0), self.z_dim).to(self.device)
        return mu + torch.exp(logvar / 2) * eps

    def sample_z_prior(self, mbsize):
        """
        Sample z ~ p(z) = N(0, I)
        """
        z = torch.randn(mbsize, self.z_dim).to(self.device)
        return z

    def sample_c_prior(self, mbsize):
        """
        Sample c ~ p(c) = Cat([0.5, 0.5])
        """
        c = torch.from_numpy(
            np.random.multinomial(1, [0.5, 0.5], mbsize).astype('float32')
        ).to(self.device)
        return c

    def forward_decoder(self, inputs, z, c):
        """
        Decode from latent space.

        Args:
            inputs: Input sequences (mbsize x seq_len)
            z: Latent vectors (mbsize x z_dim)
            c: Condition vectors (mbsize x c_dim)

        Returns:
            Logits (mbsize x seq_len x vocab_size)
        """
        return self.decoder(inputs, z, c)

    def forward(self, sequences, q_c='prior', sample_z=1):
        """
        Forward pass through encoder and decoder with teacher forcing.

        Args:
            sequences: Input sequences (mbsize x seq_len)
            q_c: 'prior' to sample c, or tensor with ground truth c's
            sample_z: 'max' to use mu, or int for number of samples

        Returns:
            (mu, logvar): From encoder
            (z, c): Input to decoder
            dec_logits: Decoder outputs
        """
        mbsize = sequences.size(0)

        # Encode
        mu, logvar = self.forward_encoder(sequences)

        if sample_z == 'max':
            z = mu
        else:
            z = self.sample_z(mu, logvar)

        # Get condition
        if isinstance(q_c, torch.Tensor):
            labels = q_c.unsqueeze(1)
            c = torch.zeros(mbsize, self.c_dim).to(self.device)
            c.scatter_(1, labels, 1)
        else:  # 'prior'
            c = self.sample_c_prior(mbsize)

        # Decode
        dec_logits = self.forward_decoder(sequences, z, c)

        return (mu, logvar), (z, c), dec_logits

    def generate_sentences(
        self,
        mbsize: int,
        z: torch.Tensor = None,
        c: torch.Tensor = None,
        eval_mode: bool = True,
        **sample_kwargs,
    ):
        """
        Generate sentences from latent space.

        Args:
            mbsize: Batch size
            z: Optional latent vectors (if None, sample from prior)
            c: Optional condition vectors (if None, sample from prior)
            eval_mode: Whether to set model to eval mode
            **sample_kwargs: Arguments for sample_G

        Returns:
            sentences: Generated sequences
            z: Latent vectors used
            c_ix: Condition indices
        """
        if z is None:
            z = self.sample_z_prior(mbsize)
        if c is None:
            c = self.sample_c_prior(mbsize)

        if eval_mode:
            self.eval()

        sentences = self.sample_G(mbsize, z, c, **sample_kwargs)

        if eval_mode:
            self.train()

        c_ix = c.argmax(dim=1)
        return sentences, z, c_ix

    def sample_G(
        self,
        mbsize: int,
        z: torch.Tensor,
        c: torch.Tensor,
        sample_mode: str = 'categorical',
        temp: float = 1.0,
        prepend_start_idx: bool = True,
        prevent_empty: bool = False,
        min_length: int = 1,
    ):
        """
        Sample from the decoder given latent code (z, c).

        Args:
            mbsize: Batch size
            z: Latent vectors (mbsize x z_dim)
            c: Condition vectors (mbsize x c_dim)
            sample_mode: 'categorical' or 'greedy'
            temp: Temperature for sampling
            prepend_start_idx: Whether to prepend <start> token
            prevent_empty: Whether to prevent empty sequences
            min_length: Minimum output length

        Returns:
            seqIx: Generated sequences (mbsize x seq_len)
        """
        seqIx = []

        # Track finished sequences
        finished = torch.zeros(mbsize, dtype=torch.bool).to(self.device)

        # Start token
        sampleIx = torch.LongTensor(mbsize).to(self.device).fill_(START_IDX)
        sampleSoftIx = None

        # Initialize RNN state
        h = self.decoder.init_hidden(z, c)
        h = h.unsqueeze(0)

        # Include start_idx in output
        if prepend_start_idx:
            seqIx.append(sampleIx)

        for i in range(self.MAX_SEQ_LEN):
            # Forward pass
            logits, h = self.decoder.forward_sample(sampleSoftIx, sampleIx, z, c, h)

            # Prevent empty sequences
            if prevent_empty and i == 0:
                large_neg = -2 * torch.abs(logits.min())
                for maskix in [PAD_IDX, START_IDX, EOS_IDX]:
                    logits[:, maskix] = large_neg

            # Sample
            if sample_mode == 'categorical':
                sampleIx = torch.distributions.Categorical(logits=logits / temp).sample()
            elif sample_mode == 'greedy':
                sampleIx = torch.argmax(logits, 1)
            else:
                raise ValueError(f'Unknown sample_mode: {sample_mode}')

            # Mask finished sequences
            sampleIx.masked_fill_(finished, PAD_IDX)
            finished[sampleIx == EOS_IDX] = True
            seqIx.append(sampleIx)

            # Check if all done
            if finished.sum() == mbsize and len(seqIx) >= min_length:
                break

        # Stack sequences
        seqIx = torch.stack(seqIx, dim=1)
        return seqIx

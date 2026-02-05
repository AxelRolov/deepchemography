"""
Vocabulary class for peptide sequences.
"""

import codecs
import logging
from typing import List, Union

import torch

from deepchemography.peptides.config import UNK_IDX, PAD_IDX, START_IDX, EOS_IDX

logger = logging.getLogger(__name__)


class PeptideVocab:
    """
    Vocabulary for converting between peptide sequences and indices.

    Peptide sequences are space-separated single-letter amino acid codes,
    e.g., "M L L L L L A L A L L A L L L A L L L"
    """

    def __init__(self, vocab_path: str = None, max_seq_len: int = 25):
        """
        Initialize vocabulary.

        Args:
            vocab_path: Path to vocab.dict file
            max_seq_len: Maximum sequence length (including special tokens)
        """
        self.max_seq_len = max_seq_len
        self.ix2word = {}
        self.word2ix = {}

        if vocab_path is not None:
            self._load_vocab(vocab_path)

        self.special_tokens = {'<unk>', '<pad>', '<start>', '<eos>'}
        self.special_tokens_ix = {
            self.word2ix.get(w, i)
            for i, w in enumerate(['<unk>', '<pad>', '<start>', '<eos>'])
        }

    def _load_vocab(self, vocab_path: str):
        """Load vocabulary from file."""
        with codecs.open(vocab_path, 'r', 'utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    word = " ".join(parts[:-1])
                    ix = int(parts[-1])
                    self.ix2word[ix] = word
                    self.word2ix[word] = ix
        logger.info(f"Loaded vocabulary with {len(self.ix2word)} tokens")

    @classmethod
    def from_default(cls, max_seq_len: int = 25):
        """Create vocabulary with default amino acid tokens."""
        vocab = cls(vocab_path=None, max_seq_len=max_seq_len)

        # Standard tokens
        tokens = [
            '<unk>', '<pad>', '<start>', '<eos>',
            'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M',
            'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'Y', 'Z'
        ]

        for ix, word in enumerate(tokens):
            vocab.ix2word[ix] = word
            vocab.word2ix[word] = ix

        return vocab

    def size(self) -> int:
        """Return vocabulary size."""
        return len(self.ix2word)

    def __len__(self) -> int:
        return self.size()

    def to_ix(self, seq: Union[str, List[str]], fix_length: bool = True) -> torch.LongTensor:
        """
        Convert sequence to indices.

        Args:
            seq: Space-separated string or list of amino acid codes
            fix_length: Whether to pad to max_seq_len

        Returns:
            LongTensor of shape (1 x seq_len)
        """
        if isinstance(seq, str):
            seq = seq.split()
        elif not isinstance(seq, list):
            raise ValueError('Only strings or lists of strings accepted.')

        # Add special tokens if not present
        if seq[0] != "<start>":
            seq = ["<start>"] + seq
        if seq[-1] != "<eos>":
            seq = seq + ["<eos>"]

        # Pad to fixed length
        if fix_length:
            if len(seq) > self.max_seq_len:
                seq = seq[:self.max_seq_len - 1] + ["<eos>"]
            else:
                num_pads = self.max_seq_len - len(seq)
                seq = seq + ["<pad>"] * num_pads

        # Convert to indices
        seq_ix = [self.word2ix.get(tok, UNK_IDX) for tok in seq]
        seq_ix = torch.LongTensor(seq_ix).view(1, -1)
        return seq_ix

    def to_word(
        self,
        seq: Union[torch.Tensor, List[int]],
        print_special_tokens: bool = True,
    ) -> List[str]:
        """
        Convert indices to amino acid codes.

        Args:
            seq: Tensor or list of indices
            print_special_tokens: Whether to include special tokens in output

        Returns:
            List of amino acid codes
        """
        if isinstance(seq, torch.Tensor):
            seq = seq.tolist()

        if not print_special_tokens:
            seq = [i for i in seq if i not in self.special_tokens_ix]

        return [self.ix2word.get(s, '<unk>') for s in seq]

    def to_string(
        self,
        seq: Union[torch.Tensor, List[int]],
        print_special_tokens: bool = False,
    ) -> str:
        """
        Convert indices to space-separated string.

        Args:
            seq: Tensor or list of indices
            print_special_tokens: Whether to include special tokens

        Returns:
            Space-separated string of amino acid codes
        """
        words = self.to_word(seq, print_special_tokens)
        return ' '.join(words)

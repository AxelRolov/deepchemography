"""
Utility classes for vocabulary and SMILES tokenization.
Extracted from ChemEidos for LSTM Autoencoder functionality.
"""
import torch
import logging
from collections import UserList, defaultdict
import pandas as pd


# Configure default logging
def setup_logging(level=logging.INFO):
    """Setup logging configuration."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


# Set up default logger
setup_logging()


class Logger(UserList):
    """Logger for tracking training metrics."""
    def __init__(self, data=None):
        super().__init__()
        self.sdata = defaultdict(list)
        for step in (data or []):
            self.append(step)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.data[key]
        if isinstance(key, slice):
            return Logger(self.data[key])
        ldata = self.sdata[key]
        if isinstance(ldata[0], dict):
            return Logger(ldata)
        return ldata

    def append(self, step_dict):
        super().append(step_dict)
        for k, v in step_dict.items():
            self.sdata[k].append(v)

    def save(self, path):
        """Save log to CSV file."""
        df = pd.DataFrame(list(self))
        df.to_csv(path, index=None)


class SpecialTokens:
    bos = '<bos>'
    eos = '<eos>'
    pad = '<pad>'
    unk = '<unk>'


class CharVocab:
    """Character-level vocabulary for SMILES tokenization."""
    
    @classmethod
    def from_data(cls, data, *args, **kwargs):
        chars = set()
        for string in data:
            chars.update(string)
        return cls(chars, *args, **kwargs)

    def __init__(self, chars, ss=SpecialTokens):
        if (ss.bos in chars) or (ss.eos in chars) or \
                (ss.pad in chars) or (ss.unk in chars):
            raise ValueError('SpecialTokens in chars')

        all_syms = sorted(list(chars)) + [ss.bos, ss.eos, ss.pad, ss.unk]

        self.ss = ss
        self.c2i = {c: i for i, c in enumerate(all_syms)}
        self.i2c = {i: c for i, c in enumerate(all_syms)}

    def __len__(self):
        return len(self.c2i)

    @property
    def bos(self):
        return self.c2i[self.ss.bos]

    @property
    def eos(self):
        return self.c2i[self.ss.eos]

    @property
    def pad(self):
        return self.c2i[self.ss.pad]

    @property
    def unk(self):
        return self.c2i[self.ss.unk]

    def char2id(self, char):
        if char not in self.c2i:
            return self.unk
        return self.c2i[char]

    def id2char(self, id):
        if id not in self.i2c:
            return self.ss.unk
        return self.i2c[id]

    def string2ids(self, string, add_bos=False, add_eos=False):
        ids = [self.char2id(c) for c in string]

        if add_bos:
            ids = [self.bos] + ids
        if add_eos:
            ids = ids + [self.eos]

        return ids

    def ids2string(self, ids, rem_bos=True, rem_eos=True):
        if len(ids) == 0:
            return ''
        if rem_bos and ids[0] == self.bos:
            ids = ids[1:]
        if rem_eos and ids[-1] == self.eos:
            ids = ids[:-1]

        string = ''.join([self.id2char(id) for id in ids])

        return string


class OneHotVocab(CharVocab):
    """Character vocabulary with one-hot encoding vectors."""
    
    def __init__(self, *args, **kwargs):
        super(OneHotVocab, self).__init__(*args, **kwargs)
        self.vectors = torch.eye(len(self.c2i))



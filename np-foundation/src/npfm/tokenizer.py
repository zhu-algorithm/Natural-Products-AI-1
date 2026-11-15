"""A small, explicit SMILES tokenizer; never silently truncate chemistry."""
from __future__ import annotations

import json
import re
from pathlib import Path


SPECIAL = ["<pad>", "<bos>", "<eos>", "<unk>", "<mask>"]
TOKEN = re.compile(r"\[[^\[\]\s]+\]|Br|Cl|Si|Se|Na|Li|Ca|Mg|Al|%\([0-9]+\)|%[0-9]{2}|@@?|[A-Za-z0-9()\.=#\-+\\/:~*$]")


class SmilesTokenizer:
    pad_id, bos_id, eos_id, unk_id, mask_id = range(5)

    def __init__(self, tokens):
        self.tokens = list(tokens)
        if self.tokens[:5] != SPECIAL or len(set(self.tokens)) != len(self.tokens):
            raise ValueError("Vocabulary must have the five ordered special tokens and no duplicates.")
        self.token_to_id = {t: i for i, t in enumerate(self.tokens)}

    def __len__(self):
        return len(self.tokens)

    @staticmethod
    def tokenize(smiles: str):
        if not isinstance(smiles, str) or not smiles or any(c.isspace() for c in smiles):
            raise ValueError("SMILES must be a nonempty string without whitespace.")
        tokens = TOKEN.findall(smiles)
        if "".join(tokens) != smiles:
            raise ValueError(f"Unsupported SMILES token syntax: {smiles[:100]}")
        return tokens

    @classmethod
    def fit(cls, smiles):
        vocabulary = {t for s in smiles for t in cls.tokenize(s)}
        return cls(SPECIAL + sorted(vocabulary - set(SPECIAL)))

    def encode(self, smiles, max_length=None, add_special_tokens=True):
        ids = [self.token_to_id.get(t, self.unk_id) for t in self.tokenize(smiles)]
        if add_special_tokens:
            ids = [self.bos_id] + ids + [self.eos_id]
        if max_length is not None and len(ids) > max_length:
            raise ValueError(f"SMILES needs {len(ids)} tokens but max_length={max_length}. No silent truncation.")
        return ids

    def decode(self, ids, skip_special_tokens=True):
        out = []
        for idx in ids:
            idx = int(idx)
            if idx == self.eos_id:
                break
            if idx < 0 or idx >= len(self.tokens):
                raise ValueError("Token id is outside the vocabulary.")
            token = self.tokens[idx]
            if skip_special_tokens and token in SPECIAL:
                continue
            out.append(token)
        return "".join(out)

    def save(self, path):
        Path(path).write_text(json.dumps({"format_version": 1, "tokens": self.tokens}, indent=2) + "\n")

    @classmethod
    def load(cls, path):
        value = json.loads(Path(path).read_text())
        return cls(value["tokens"] if isinstance(value, dict) else value)

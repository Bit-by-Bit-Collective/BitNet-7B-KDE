
# scripts/data.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as pds


def _to_python(value: Any) -> Any:
    """Best-effort conversion from Arrow scalars to native Python types."""
    if isinstance(value, pa.Scalar):
        return value.as_py()
    return value


def _ensure_list(x: Any) -> List[Any]:
    """Normalize a possibly-null/pyarrow list-like to a Python list."""
    if x is None:
        return []
    if isinstance(x, pa.Array):
        return list(x.to_pylist())
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


class KDTraceDataset(Dataset):
    """
    KD traces dataset (Top-K + Other) grouped into per-prompt sequences.

    Expected Parquet schema columns (lists are per-token-step):
      - prompt_idx: int
      - step_idx:   int
      - teacher_sample_id: int
      - topk_ids: List[int]
      - topk_logprobs: List[float]   (log probabilities from teacher)
      - other_logprob: float         (log mass of "all others")
      - is_struct_token: bool        (e.g., JSON punctuation)

    The dataset groups rows by prompt_idx, sorts by step_idx, and returns
    a sequence of tokens per prompt. Each __getitem__ returns a dict of tensors:

      {
        'input_ids':        LongTensor [T]
        'teacher_sample_id': LongTensor [T]
        'topk_ids':         LongTensor [T, K]
        'topk_logprobs':    FloatTensor [T, K]
        'other_logprob':    FloatTensor [T]
        'is_struct_token':  BoolTensor [T]
      }

    Note: sequences are truncated to max_seq_len; top-k is truncated/padded to max_topk.
    """

    def __init__(
        self,
        parquet_path_or_dir: Union[str, Path],
        max_seq_len: int = 512,
        max_topk: int = 20,
        min_tokens_per_sequence: int = 2,
    ):
        super().__init__()
        self.max_seq_len = int(max_seq_len)
        self.max_topk = int(max_topk)

        path = Path(parquet_path_or_dir)
        if not path.exists():
            raise FileNotFoundError(f"KDTraceDataset: path not found: {path}")

        # Load one or multiple parquet files
        if path.is_dir():
            ds = pds.dataset(path, format="parquet")
            table = ds.to_table()
        else:
            table = pq.read_table(path)

        data = table.to_pydict()

        required_cols = [
            "prompt_idx",
            "step_idx",
            "teacher_sample_id",
            "topk_ids",
            "topk_logprobs",
            "other_logprob",
            "is_struct_token",
        ]
        for col in required_cols:
            if col not in data:
                raise ValueError(f"Missing required column '{col}' in {path}")

        # Group steps by prompt_idx
        from collections import defaultdict

        sequences = defaultdict(list)
        n_rows = len(data["prompt_idx"])
        for i in range(n_rows):
            prompt_idx = _to_python(data["prompt_idx"][i])
            sequences[prompt_idx].append(
                {
                    "step_idx": _to_python(data["step_idx"][i]),
                    "teacher_sample_id": _to_python(data["teacher_sample_id"][i]),
                    "topk_ids": _ensure_list(_to_python(data["topk_ids"][i])),
                    "topk_logprobs": _ensure_list(_to_python(data["topk_logprobs"][i])),
                    "other_logprob": float(_to_python(data["other_logprob"][i])),
                    "is_struct_token": bool(_to_python(data["is_struct_token"][i])),
                }
            )

        # Convert to per-sequence list, sort by step_idx, filter short, and truncate
        self.sequences: List[List[Dict[str, Any]]] = []
        kept = 0
        for _, steps in sequences.items():
            steps.sort(key=lambda x: x["step_idx"])
            if len(steps) >= min_tokens_per_sequence:
                # Truncate to max_seq_len
                if len(steps) > self.max_seq_len:
                    steps = steps[: self.max_seq_len]
                self.sequences.append(steps)
                kept += 1

        print(f"Loaded {kept} sequences from {path} (rows={n_rows})")

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        seq = self.sequences[idx]

        # Extract raw lists
        teacher_ids: List[int] = []
        topk_ids_list: List[List[int]] = []
        topk_lps_list: List[List[float]] = []
        other_lps: List[float] = []
        is_struct_flags: List[bool] = []

        for step in seq:
            # Teacher next-token id (will be used with next-token alignment in loss)
            teacher_ids.append(int(step["teacher_sample_id"]))

            # Top-K lists — truncate to K and align lengths
            tk_ids = list(step["topk_ids"])[: self.max_topk]
            tk_lps = list(step["topk_logprobs"])[: self.max_topk]

            # Ensure equal length before padding
            if len(tk_lps) < len(tk_ids):
                tk_lps += [-float("inf")] * (len(tk_ids) - len(tk_lps))
            if len(tk_ids) < len(tk_lps):
                tk_ids += [0] * (len(tk_lps) - len(tk_ids))

            # Pad to K
            if len(tk_ids) < self.max_topk:
                pad_k = self.max_topk - len(tk_ids)
                tk_ids += [0] * pad_k
                tk_lps += [-float("inf")] * pad_k

            topk_ids_list.append(tk_ids[: self.max_topk])
            topk_lps_list.append(tk_lps[: self.max_topk])

            # Other logprob and struct flag
            other_lps.append(float(step["other_logprob"]))
            is_struct_flags.append(bool(step["is_struct_token"]))

        # Build tensors
        T = len(teacher_ids)
        item = {
            "input_ids": torch.tensor(teacher_ids, dtype=torch.long),  # same as teacher for convenience
            "teacher_sample_id": torch.tensor(teacher_ids, dtype=torch.long),
            "topk_ids": torch.tensor(topk_ids_list, dtype=torch.long),  # [T, K]
            "topk_logprobs": torch.tensor(topk_lps_list, dtype=torch.float),  # [T, K]
            "other_logprob": torch.tensor(other_lps, dtype=torch.float),  # [T]
            "is_struct_token": torch.tensor(is_struct_flags, dtype=torch.bool),  # [T]
        }
        return item


def make_collate_fn(pad_token_id: int):
    """
    Factory returning a collate_fn that:
      - pads time dimension to longest sequence in batch
      - pads top-k along time only (K stays constant)
      - provides attention_mask (1=token, 0=pad)
      - uses -100 for padded labels (teacher_sample_id) for CE ignore_index
    """

    def _collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        max_len = max(x["input_ids"].size(0) for x in batch)

        input_ids: List[torch.Tensor] = []
        teacher_sample_ids: List[torch.Tensor] = []
        topk_ids: List[torch.Tensor] = []
        topk_logprobs: List[torch.Tensor] = []
        other_logprob: List[torch.Tensor] = []
        is_struct_token: List[torch.Tensor] = []
        attention_mask: List[torch.Tensor] = []

        for x in batch:
            T = x["input_ids"].size(0)
            pad = max_len - T

            input_ids.append(F.pad(x["input_ids"], (0, pad), value=pad_token_id))
            teacher_sample_ids.append(F.pad(x["teacher_sample_id"], (0, pad), value=-100))

            # Pad time dimension for [T, K]
            topk_ids.append(F.pad(x["topk_ids"], (0, 0, 0, pad), value=0))
            topk_logprobs.append(F.pad(x["topk_logprobs"], (0, 0, 0, pad), value=-float("inf")))

            other_logprob.append(F.pad(x["other_logprob"], (0, pad), value=-float("inf")))
            is_struct_token.append(F.pad(x["is_struct_token"], (0, pad), value=False))

            attn = torch.ones(T, dtype=torch.long)
            attention_mask.append(F.pad(attn, (0, pad), value=0))

        return {
            "input_ids": torch.stack(input_ids),
            "teacher_sample_id": torch.stack(teacher_sample_ids),
            "topk_ids": torch.stack(topk_ids),
            "topk_logprobs": torch.stack(topk_logprobs),
            "other_logprob": torch.stack(other_logprob),
            "is_struct_token": torch.stack(is_struct_token),
            "attention_mask": torch.stack(attention_mask),
        }

    return _collate


def make_dataloader(
    parquet_path_or_dir: Union[str, Path],
    pad_token_id: int,
    batch_size: int = 4,
    shuffle: bool = True,
    max_seq_len: int = 256,
    max_topk: int = 20,
    num_workers: int = 0,
    pin_memory: Optional[bool] = None,
) -> Tuple[KDTraceDataset, DataLoader]:
    """
    Convenience wrapper: build KDTraceDataset + DataLoader with a correct collate_fn.
    """
    dataset = KDTraceDataset(parquet_path_or_dir, max_seq_len=max_seq_len, max_topk=max_topk)

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=make_collate_fn(pad_token_id),
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    return dataset, loader


if __name__ == "__main__":
    # Tiny smoke test (expects a valid shard)
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", type=str, required=True, help="Path to parquet file or directory")
    ap.add_argument("--pad_token_id", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=2)
    args = ap.parse_args()

    ds, dl = make_dataloader(
        args.parquet,
        pad_token_id=args.pad_token_id,
        batch_size=args.batch_size,
        shuffle=False,
        max_seq_len=64,
        max_topk=20,
    )
    print(f"Dataset sequences: {len(ds)}")
    batch = next(iter(dl))
    for k, v in batch.items():
        print(f"{k:>16}: {tuple(v.shape)} {v.dtype}")

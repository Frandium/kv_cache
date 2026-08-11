import os
import csv
import json
import glob
import torch
import tiktoken
import argparse
import numpy as np

from torch.utils.data import Dataset, DataLoader, DistributedSampler, IterableDataset, get_worker_info

TEST_COUNT = 10000

class Tokenized_data(Dataset):
    def __init__(self, window_size, is_test=False, start_from=0, folder_prefix=None) -> None:
        super().__init__()
        if folder_prefix is None:
            domain_name = 'dclm_every_4096'
            model_name = 'qwen'
            self.folder_prefix = f"/mnt/workspace/let-moe_predictor/data/{domain_name}_{model_name}"
        else:
            self.folder_prefix = folder_prefix
        self.domain_files = sorted(os.listdir(self.folder_prefix))
        assert len(self.domain_files) > 0, f"empty folder: {self.folder_prefix}"

        self.cur_file_idx = 0
        with open(f'{self.folder_prefix}/{self.domain_files[0]}', 'r') as f:
            self.file_texts = f.readlines()

        self.lines_per_file = len(self.file_texts)
        if self.lines_per_file <= 0:
            self.lines_per_file = 1

        self.window_size = window_size
        self.is_test = is_test
        self.start_from = start_from
        self.pad_id = 0

        first_line = self.file_texts[0].strip()
        self.token_id_mode = all(ch.isdigit() or ch.isspace() for ch in first_line)

    def __len__(self):
        total = len(self.domain_files) * self.lines_per_file - self.start_from
        if self.is_test:
            return min(10, total)
        return total

    def __getitem__(self, index):
        index += self.start_from
        file_idx = index // self.lines_per_file
        line_idx = index % self.lines_per_file

        if file_idx != self.cur_file_idx:
            with open(f'{self.folder_prefix}/{self.domain_files[file_idx]}', 'r') as f:
                self.file_texts = f.readlines()
            self.cur_file_idx = file_idx

        if line_idx >= len(self.file_texts):
            line_idx = len(self.file_texts) - 1

        line = self.file_texts[line_idx].strip()

        if self.token_id_mode:
            tokens = [int(x) for x in line.split() if x.strip() != ""]
        else:
            line = line.replace('<|', '').replace('|>', '')
            tokens = tiktoken.get_encoding('r50k_base').encode(line)

        source = tokens[:self.window_size]
        target = tokens[1:self.window_size + 1]

        # 分别补齐，避免 target 比 source 少 1 的情况导致 batch stack 报错
        if len(source) < self.window_size:
            source += [self.pad_id] * (self.window_size - len(source))
        if len(target) < self.window_size:
            target += [self.pad_id] * (self.window_size - len(target))

        source = torch.tensor(source).long()
        target = torch.tensor(target).long()
        return source, target, 0


class StreamingTokenizedData(IterableDataset):
    def __init__(
        self,
        window_size,
        is_test=False,
        folder_prefix=None,
        repeat=False,
        max_samples=None,
        pad_id=0,
        data_mode=None,
        file_glob=None,
        text_key=None,
        token_key=None,
        parquet_batch_size=2048,
    ) -> None:
        super().__init__()
        if folder_prefix is None:
            domain_name = "dclm_every_4096"
            model_name = "qwen"
            self.folder_prefix = f"/mnt/workspace/let-moe_predictor/data/{domain_name}_{model_name}"
        else:
            self.folder_prefix = folder_prefix

        self.file_glob = file_glob or os.environ.get("DATA_FILE_GLOB", "*")
        self.domain_files = sorted(
            [p for p in glob.glob(os.path.join(self.folder_prefix, self.file_glob)) if os.path.isfile(p)]
        )
        if len(self.domain_files) == 0:
            self.domain_files = sorted(
                [
                    os.path.join(self.folder_prefix, fn)
                    for fn in os.listdir(self.folder_prefix)
                    if os.path.isfile(os.path.join(self.folder_prefix, fn))
                ]
            )
        assert len(self.domain_files) > 0, f"empty folder: {self.folder_prefix}"

        self.window_size = int(window_size)
        self.is_test = bool(is_test)
        self.repeat = bool(repeat)
        self.max_samples = max_samples
        self.pad_id = int(pad_id)
        self.parquet_batch_size = int(parquet_batch_size)
        self.data_mode = (data_mode or os.environ.get("DATA_MODE", "auto")).lower()
        if self.data_mode not in {"auto", "text", "tokens"}:
            raise ValueError(f"Unsupported data_mode={self.data_mode}, use auto|text|tokens")

        self.text_key = text_key or os.environ.get("DATA_TEXT_KEY")
        self.token_key = token_key or os.environ.get("DATA_TOKEN_KEY")
        self._default_text_keys = [k for k in [self.text_key, "text", "content", "raw_text", "prompt"] if k]
        self._default_token_keys = [k for k in [self.token_key, "token_ids", "input_ids", "ids", "tokens"] if k]
        self._enc = None

    def _ensure_tokenizer(self):
        if self._enc is None:
            self._enc = tiktoken.get_encoding("r50k_base")
        return self._enc

    def _parse_token_line(self, s: str):
        s = s.strip()
        if not s:
            return None
        parts = s.split()
        if parts:
            try:
                return [int(x) for x in parts]
            except ValueError:
                pass
        if s.startswith("[") and s.endswith("]"):
            try:
                arr = json.loads(s)
                if isinstance(arr, list):
                    return [int(x) for x in arr]
            except Exception:
                return None
        return None

    def _tokenize_text(self, text: str):
        text = text.strip()
        if not text:
            return None
        enc = self._ensure_tokenizer()
        text = text.replace("<|", "").replace("|>", "")
        return enc.encode(text)

    def _tokens_from_value(self, value):
        if value is None:
            return None
        if isinstance(value, (list, tuple, np.ndarray)):
            try:
                return [int(x) for x in value]
            except Exception:
                return None
        if isinstance(value, (int, np.integer)):
            return [int(value)]
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="ignore")
        if isinstance(value, str):
            if self.data_mode == "tokens":
                return self._parse_token_line(value)
            if self.data_mode == "text":
                return self._tokenize_text(value)
            tok = self._parse_token_line(value)
            if tok is not None:
                return tok
            return self._tokenize_text(value)
        return None

    def _record_to_tokens(self, record):
        if isinstance(record, (str, bytes, list, tuple, np.ndarray, int, np.integer)):
            return self._tokens_from_value(record)

        if isinstance(record, dict):
            if self.data_mode in {"auto", "tokens"}:
                for k in self._default_token_keys:
                    if k in record:
                        tok = self._tokens_from_value(record[k])
                        if tok:
                            return tok

            if self.data_mode in {"auto", "text"}:
                for k in self._default_text_keys:
                    if k in record:
                        tok = self._tokens_from_value(record[k])
                        if tok:
                            return tok

            if len(record) == 1:
                v = next(iter(record.values()))
                return self._tokens_from_value(v)
            return None

        return None

    def _iter_txt_records(self, fp):
        with open(fp, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.strip():
                    yield line

    def _iter_jsonl_records(self, fp):
        with open(fp, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    yield line

    def _iter_parquet_records(self, fp):
        try:
            import pyarrow.parquet as pq
            pf = pq.ParquetFile(fp)
            cols = list(dict.fromkeys(self._default_token_keys + self._default_text_keys)) or None
            for batch in pf.iter_batches(batch_size=self.parquet_batch_size, columns=cols):
                data = batch.to_pydict()
                keys = list(data.keys())
                if not keys:
                    continue
                size = len(data[keys[0]])
                for i in range(size):
                    yield {k: data[k][i] for k in keys}
            return
        except ImportError:
            pass

        try:
            import pandas as pd
            cols = list(dict.fromkeys(self._default_token_keys + self._default_text_keys)) or None
            df = pd.read_parquet(fp, columns=cols)
            for rec in df.to_dict(orient="records"):
                yield rec
            return
        except ImportError as e:
            raise ImportError("Reading parquet requires pyarrow or pandas. Please install one of them.") from e

    def _iter_file_records(self, fp):
        ext = os.path.splitext(fp)[1].lower()
        if ext in {".txt", ".text", ".log", ".csv"}:
            yield from self._iter_txt_records(fp)
            return
        if ext in {".jsonl", ".json"}:
            yield from self._iter_jsonl_records(fp)
            return
        if ext in {".parquet", ".pq"}:
            yield from self._iter_parquet_records(fp)
            return
        yield from self._iter_txt_records(fp)

    def _to_sample(self, tokens):
        source = tokens[: self.window_size]
        target = tokens[1 : self.window_size + 1]

        # 分别补齐，避免 255/256 长度不一致
        if len(source) < self.window_size:
            source += [self.pad_id] * (self.window_size - len(source))
        if len(target) < self.window_size:
            target += [self.pad_id] * (self.window_size - len(target))

        source = torch.tensor(source, dtype=torch.long)
        target = torch.tensor(target, dtype=torch.long)
        return source, target, 0

    def __iter__(self):
        worker = get_worker_info()
        if worker is None:
            files = self.domain_files
        else:
            files = self.domain_files[worker.id :: worker.num_workers]

        if not files:
            return

        yielded = 0
        while True:
            for fp in files:
                for record in self._iter_file_records(fp):
                    tokens = self._record_to_tokens(record)
                    if not tokens:
                        continue
                    yield self._to_sample(tokens)
                    yielded += 1
                    if self.max_samples is not None and yielded >= self.max_samples:
                        return
                    if self.is_test and yielded >= 1000:
                        return
            if not self.repeat:
                return

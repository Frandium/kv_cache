#!/usr/bin/env python3
"""Mine C-4-2 and C-8 pair-of-pairs from line-delimited DCLM text.

The implementation is model-free after tokenization. Event keys are partitioned
into temporary binary buckets so counting remains exact without a giant Python
dictionary.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import math
import random
import shutil
import time
from collections import OrderedDict, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
from transformers import AutoTokenizer


TOKEN_BITS = 20
TOKEN_MASK = (1 << TOKEN_BITS) - 1
PAIR_GAP_BITS = 2
PAIR_GAP_MASK = (1 << PAIR_GAP_BITS) - 1
LEVEL1_MAX_SPAN = 4
LEVEL2_MAX_SPAN = 8
LEVEL2_GAP_BUCKETS = 4


@dataclass
class Document:
    doc_id: int
    relative_path: str
    line_number: int
    text: str


@dataclass
class Candidate:
    key: int
    left: int
    right: int
    gap: int
    train_support: int
    valid_support: int
    npmi: float
    heldout_log_rate_diff: float
    kind: str = "hetero"
    document_count: int = 0
    parent_reuse: int = 0


@dataclass(frozen=True)
class Occurrence:
    pattern_id: int
    start: int
    end: int


class RunLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, message: str) -> None:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {message}"
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


class ProgressReporter:
    def __init__(
        self,
        logger: RunLogger,
        phase: str,
        total_documents: int,
        every_documents: int,
    ) -> None:
        self.logger = logger
        self.phase = phase
        self.total_documents = total_documents
        self.every_documents = max(1, every_documents)
        self.start_time = time.perf_counter()
        self.last_time = self.start_time
        self.last_documents = 0
        self.last_tokens = 0

    def update(
        self,
        documents: int,
        tokens: int,
        event_text: str = "",
        force: bool = False,
    ) -> None:
        if force and documents == self.last_documents and tokens == self.last_tokens:
            return
        if not force and documents - self.last_documents < self.every_documents:
            return
        now = time.perf_counter()
        elapsed = max(now - self.start_time, 1e-9)
        interval = max(now - self.last_time, 1e-9)
        recent_docs_s = (documents - self.last_documents) / interval
        recent_tokens_s = (tokens - self.last_tokens) / interval
        average_docs_s = documents / elapsed
        average_tokens_s = tokens / elapsed
        remaining = max(self.total_documents - documents, 0)
        eta = remaining / recent_docs_s if recent_docs_s > 0 else float("inf")
        eta_text = format_duration(eta) if math.isfinite(eta) else "unknown"
        suffix = f", {event_text}" if event_text else ""
        self.logger.log(
            f"{self.phase}: docs={documents:,}/{self.total_documents:,}, "
            f"tokens={tokens:,}, recent={recent_docs_s:.1f} docs/s "
            f"({recent_tokens_s:,.0f} tok/s), average={average_docs_s:.1f} docs/s "
            f"({average_tokens_s:,.0f} tok/s), ETA={eta_text}{suffix}"
        )
        self.last_time = now
        self.last_documents = documents
        self.last_tokens = tokens


class BucketWriter:
    """Append uint64 keys into hash-partitioned files with bounded open handles."""

    def __init__(self, root: Path, num_buckets: int, max_open_files: int = 24) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.num_buckets = num_buckets
        self.max_open_files = max_open_files
        self.handles: OrderedDict[int, object] = OrderedDict()
        self.events_written = 0

    def _path(self, bucket: int) -> Path:
        return self.root / f"bucket_{bucket:04d}.bin"

    def _handle(self, bucket: int):
        handle = self.handles.pop(bucket, None)
        if handle is None:
            if len(self.handles) >= self.max_open_files:
                _, old_handle = self.handles.popitem(last=False)
                old_handle.close()
            handle = self._path(bucket).open("ab")
        self.handles[bucket] = handle
        return handle

    def write(self, keys: np.ndarray) -> None:
        keys = np.asarray(keys, dtype=np.uint64)
        if keys.size == 0:
            return
        bucket_ids = np.remainder(keys, self.num_buckets).astype(np.int32)
        order = np.argsort(bucket_ids, kind="stable")
        sorted_buckets = bucket_ids[order]
        starts = np.r_[0, np.flatnonzero(np.diff(sorted_buckets)) + 1]
        ends = np.r_[starts[1:], keys.size]
        for start, end in zip(starts, ends):
            bucket = int(sorted_buckets[start])
            keys[order[start:end]].tofile(self._handle(bucket))
        self.events_written += int(keys.size)

    def close(self) -> None:
        for handle in self.handles.values():
            handle.close()
        self.handles.clear()


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds}s"
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m{seconds:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


def stable_train_split(relative_path: str, line_number: int, train_fraction: float) -> bool:
    value = f"{relative_path}:{line_number}".encode("utf-8")
    digest = hashlib.blake2b(value, digest_size=8).digest()
    unit = int.from_bytes(digest, "little") / float(1 << 64)
    return unit < train_fraction


def parse_document_line(line: str) -> Optional[str]:
    stripped = line.strip()
    if not stripped:
        return None
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        return stripped
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for field in ("text", "content", "document"):
            text = value.get(field)
            if isinstance(text, str):
                return text
    return None


def iter_documents(
    data_dir: Path,
    max_documents: int,
    max_documents_per_file: int,
    seed: int,
) -> Iterator[Document]:
    files = sorted(data_dir.rglob("*.txt"))
    rng = random.Random(seed)
    rng.shuffle(files)
    doc_id = 0
    for path in files:
        relative_path = str(path.relative_to(data_dir))
        accepted_from_file = 0
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line_number, line in enumerate(handle, start=1):
                text = parse_document_line(line)
                if not text:
                    continue
                yield Document(doc_id, relative_path, line_number, text)
                doc_id += 1
                accepted_from_file += 1
                if doc_id >= max_documents:
                    return
                if accepted_from_file >= max_documents_per_file:
                    break


def encode_level1_keys(tokens: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:
    tokens = np.asarray(tokens, dtype=np.uint64)
    arrays: List[np.ndarray] = []
    opportunities = np.zeros(3, dtype=np.uint64)
    for distance in range(1, LEVEL1_MAX_SPAN):
        if tokens.size <= distance:
            arrays.append(np.empty(0, dtype=np.uint64))
            continue
        left = tokens[:-distance]
        right = tokens[distance:]
        keys = (left << (TOKEN_BITS + PAIR_GAP_BITS)) | (right << PAIR_GAP_BITS)
        keys |= np.uint64(distance - 1)
        arrays.append(keys)
        opportunities[distance - 1] = keys.size
    return arrays, opportunities


def decode_level1_key(key: int) -> Tuple[int, int, int]:
    gap = key & PAIR_GAP_MASK
    right = (key >> PAIR_GAP_BITS) & TOKEN_MASK
    left = key >> (TOKEN_BITS + PAIR_GAP_BITS)
    return left, right, gap


def level2_gap_bucket(gap_tokens: int) -> int:
    if gap_tokens <= 0:
        return 0
    if gap_tokens == 1:
        return 1
    if gap_tokens <= 3:
        return 2
    return 3


def encode_level2_key(left_id: int, right_id: int, gap: int, id_bits: int) -> int:
    return (left_id << (id_bits + 3)) | (right_id << 3) | gap


def decode_level2_key(key: int, id_bits: int) -> Tuple[int, int, int]:
    gap = key & 0b111
    right = (key >> 3) & ((1 << id_bits) - 1)
    left = key >> (id_bits + 3)
    return left, right, gap


def load_bucket_counts(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    if not path.exists() or path.stat().st_size == 0:
        return np.empty(0, dtype=np.uint64), np.empty(0, dtype=np.int64)
    values = np.fromfile(path, dtype=np.uint64)
    keys, counts = np.unique(values, return_counts=True)
    return keys, counts.astype(np.int64, copy=False)


def lookup_counts(
    query_keys: np.ndarray,
    reference_keys: np.ndarray,
    reference_counts: np.ndarray,
) -> np.ndarray:
    output = np.zeros(query_keys.size, dtype=np.int64)
    if query_keys.size == 0 or reference_keys.size == 0:
        return output
    positions = np.searchsorted(reference_keys, query_keys)
    in_bounds = positions < reference_keys.size
    safe_positions = np.minimum(positions, reference_keys.size - 1)
    matched = in_bounds & (reference_keys[safe_positions] == query_keys)
    output[matched] = reference_counts[positions[matched]]
    return output


def support_bucket(support: int) -> int:
    return max(0, int(support).bit_length() - 1)


def push_beam(
    heaps: Dict[Tuple[str, int], list],
    candidate: Candidate,
    beam_size: int,
) -> None:
    bucket = (candidate.kind, support_bucket(candidate.train_support))
    entry = (
        candidate.npmi,
        candidate.train_support,
        -candidate.heldout_log_rate_diff,
        candidate.key,
        candidate,
    )
    heap = heaps[bucket]
    if beam_size <= 0 or len(heap) < beam_size:
        heapq.heappush(heap, entry)
    elif entry[:4] > heap[0][:4]:
        heapq.heapreplace(heap, entry)


def flatten_beams(heaps: Dict[Tuple[str, int], list]) -> List[Candidate]:
    candidates = [entry[-1] for heap in heaps.values() for entry in heap]
    candidates.sort(key=lambda item: (-item.train_support, -item.npmi, item.key))
    return candidates


def mine_level1(
    event_root: Path,
    num_buckets: int,
    unigram_counts: np.ndarray,
    token_totals: np.ndarray,
    opportunities: np.ndarray,
    min_train_support: int,
    min_valid_support: int,
    min_npmi: float,
    max_log_rate_diff: float,
    beam_size: int,
    logger: RunLogger,
) -> Tuple[List[Candidate], dict]:
    heaps: Dict[Tuple[str, int], list] = defaultdict(list)
    stats = {
        "unique_train_candidates": 0,
        "pass_train_support": 0,
        "pass_valid_support": 0,
        "pass_npmi": 0,
        "pass_stability": 0,
    }
    train_tokens = max(int(token_totals[0]), 1)
    valid_tokens = max(int(token_totals[1]), 1)

    for bucket in range(num_buckets):
        train_keys, train_counts = load_bucket_counts(
            event_root / "train" / f"bucket_{bucket:04d}.bin"
        )
        valid_keys, valid_counts = load_bucket_counts(
            event_root / "valid" / f"bucket_{bucket:04d}.bin"
        )
        stats["unique_train_candidates"] += int(train_keys.size)
        support_mask = train_counts >= min_train_support
        stats["pass_train_support"] += int(support_mask.sum())
        train_keys = train_keys[support_mask]
        train_counts = train_counts[support_mask]
        if train_keys.size == 0:
            continue
        matched_valid = lookup_counts(train_keys, valid_keys, valid_counts)
        valid_mask = matched_valid >= min_valid_support
        stats["pass_valid_support"] += int(valid_mask.sum())
        train_keys = train_keys[valid_mask]
        train_counts = train_counts[valid_mask]
        matched_valid = matched_valid[valid_mask]
        if train_keys.size == 0:
            continue

        gaps = (train_keys & PAIR_GAP_MASK).astype(np.int64)
        rights = ((train_keys >> PAIR_GAP_BITS) & TOKEN_MASK).astype(np.int64)
        lefts = (train_keys >> (TOKEN_BITS + PAIR_GAP_BITS)).astype(np.int64)
        joint = train_counts / np.maximum(opportunities[0, gaps], 1)
        left_prob = unigram_counts[0, lefts] / train_tokens
        right_prob = unigram_counts[0, rights] / train_tokens
        ratio = joint / np.maximum(left_prob * right_prob, 1e-300)
        denominator = np.maximum(-np.log(np.minimum(joint, 1 - 1e-15)), 1e-15)
        npmi_values = np.log(np.maximum(ratio, 1e-300)) / denominator
        npmi_mask = npmi_values >= min_npmi
        stats["pass_npmi"] += int(npmi_mask.sum())

        train_keys = train_keys[npmi_mask]
        train_counts = train_counts[npmi_mask]
        matched_valid = matched_valid[npmi_mask]
        lefts = lefts[npmi_mask]
        rights = rights[npmi_mask]
        gaps = gaps[npmi_mask]
        npmi_values = npmi_values[npmi_mask]
        if train_keys.size == 0:
            continue

        train_rates = train_counts / train_tokens
        valid_rates = matched_valid / valid_tokens
        rate_diff = np.abs(np.log(np.maximum(valid_rates, 1e-300) / np.maximum(train_rates, 1e-300)))
        stable_mask = rate_diff <= max_log_rate_diff
        stats["pass_stability"] += int(stable_mask.sum())
        for key, left, right, gap, train_count, valid_count, score, stability in zip(
            train_keys[stable_mask],
            lefts[stable_mask],
            rights[stable_mask],
            gaps[stable_mask],
            train_counts[stable_mask],
            matched_valid[stable_mask],
            npmi_values[stable_mask],
            rate_diff[stable_mask],
        ):
            candidate = Candidate(
                key=int(key),
                left=int(left),
                right=int(right),
                gap=int(gap),
                train_support=int(train_count),
                valid_support=int(valid_count),
                npmi=float(score),
                heldout_log_rate_diff=float(stability),
            )
            push_beam(heaps, candidate, beam_size)
        if (bucket + 1) % max(1, num_buckets // 8) == 0:
            logger.log(f"C-4-2 counting: processed {bucket + 1}/{num_buckets} buckets")

    retained = flatten_beams(heaps)
    stats["retained_after_beam"] = len(retained)
    return retained, stats


def active_occurrences(
    tokens: np.ndarray,
    active_keys_sorted: np.ndarray,
    active_ids_sorted: np.ndarray,
) -> List[Occurrence]:
    occurrences: List[Occurrence] = []
    if active_keys_sorted.size == 0:
        return occurrences
    arrays, _ = encode_level1_keys(tokens)
    for gap, keys in enumerate(arrays):
        if keys.size == 0:
            continue
        positions = np.searchsorted(active_keys_sorted, keys)
        in_bounds = positions < active_keys_sorted.size
        safe_positions = np.minimum(positions, active_keys_sorted.size - 1)
        matched = in_bounds & (active_keys_sorted[safe_positions] == keys)
        starts = np.flatnonzero(matched)
        pattern_ids = active_ids_sorted[positions[matched]]
        distance = gap + 1
        occurrences.extend(
            Occurrence(int(pattern_id), int(start), int(start + distance))
            for start, pattern_id in zip(starts, pattern_ids)
        )
    occurrences.sort(key=lambda item: (item.start, item.end, item.pattern_id))
    return occurrences


def permute_occurrence_identities(
    occurrences: Sequence[Occurrence],
    rng: np.random.Generator,
) -> List[Occurrence]:
    permuted = list(occurrences)
    by_span: Dict[int, List[int]] = defaultdict(list)
    for index, occurrence in enumerate(occurrences):
        by_span[occurrence.end - occurrence.start + 1].append(index)
    for indices in by_span.values():
        pattern_ids = np.asarray([occurrences[index].pattern_id for index in indices], dtype=np.int64)
        rng.shuffle(pattern_ids)
        for index, pattern_id in zip(indices, pattern_ids):
            original = occurrences[index]
            permuted[index] = Occurrence(int(pattern_id), original.start, original.end)
    permuted.sort(key=lambda item: (item.start, item.end, item.pattern_id))
    return permuted


def compose_occurrences(
    occurrences: Sequence[Occurrence],
    document_length: int,
    id_bits: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    by_start: List[List[Occurrence]] = [[] for _ in range(document_length)]
    for occurrence in occurrences:
        by_start[occurrence.start].append(occurrence)

    keys: List[int] = []
    left_ids: List[int] = []
    right_ids: List[int] = []
    gaps: List[int] = []
    parent_starts: List[int] = []
    parent_ends: List[int] = []
    seen = set()

    for left in occurrences:
        latest_start = min(document_length - 1, left.start + LEVEL2_MAX_SPAN - 1)
        for right_start in range(left.end + 1, latest_start + 1):
            for right in by_start[right_start]:
                if right.end - left.start + 1 > LEVEL2_MAX_SPAN:
                    continue
                gap = level2_gap_bucket(right.start - left.end - 1)
                key = encode_level2_key(left.pattern_id, right.pattern_id, gap, id_bits)
                signature = (key, left.start, right.end)
                if signature in seen:
                    continue
                seen.add(signature)
                keys.append(key)
                left_ids.append(left.pattern_id)
                right_ids.append(right.pattern_id)
                gaps.append(gap)
                parent_starts.append(left.start)
                parent_ends.append(right.end)

    return (
        np.asarray(keys, dtype=np.uint64),
        np.asarray(left_ids, dtype=np.int64),
        np.asarray(right_ids, dtype=np.int64),
        np.asarray(gaps, dtype=np.int64),
        np.asarray(parent_starts, dtype=np.int32),
        np.asarray(parent_ends, dtype=np.int32),
    )


def tokenize_and_spool_level1(
    args: argparse.Namespace,
    tokenizer,
    output_dir: Path,
    logger: RunLogger,
) -> dict:
    cache_dir = output_dir / "cache"
    event_root = output_dir / "_events" / "c4"
    cache_dir.mkdir(parents=True, exist_ok=True)
    train_writer = BucketWriter(event_root / "train", args.num_buckets)
    valid_writer = BucketWriter(event_root / "valid", args.num_buckets)
    token_path = cache_dir / "tokens.bin"
    metadata_path = cache_dir / "documents.jsonl"

    token_totals = np.zeros(2, dtype=np.uint64)
    unigram_counts = np.zeros((2, len(tokenizer)), dtype=np.uint64)
    opportunities = np.zeros((2, 3), dtype=np.uint64)
    offsets = [0]
    splits: List[int] = []
    documents_processed = 0
    max_length_documents = 0
    progress = ProgressReporter(
        logger, "tokenize+C-4-2 spool", args.max_documents, args.progress_every
    )

    document_iterator = iter_documents(
        Path(args.data_dir),
        args.max_documents,
        args.max_documents_per_file,
        args.seed,
    )
    pending: List[Document] = []

    with token_path.open("wb") as token_handle, metadata_path.open("w", encoding="utf-8") as metadata_handle:
        while True:
            pending.clear()
            try:
                for _ in range(args.tokenizer_batch_size):
                    pending.append(next(document_iterator))
            except StopIteration:
                pass
            if not pending:
                break

            encoded = tokenizer(
                [document.text for document in pending],
                add_special_tokens=False,
                truncation=True,
                max_length=args.max_tokens_per_document,
                padding=False,
                return_attention_mask=False,
            )["input_ids"]
            batch_events: List[List[np.ndarray]] = [[], []]

            for document, token_list in zip(pending, encoded):
                if not token_list:
                    continue
                tokens = np.asarray(token_list, dtype=np.uint32)
                if tokens.size >= args.max_tokens_per_document:
                    max_length_documents += 1
                is_train = stable_train_split(
                    document.relative_path, document.line_number, args.train_fraction
                )
                split = 0 if is_train else 1
                tokens.tofile(token_handle)
                offsets.append(offsets[-1] + int(tokens.size))
                splits.append(split)
                np.add.at(unigram_counts[split], tokens, 1)
                token_totals[split] += tokens.size
                arrays, doc_opportunities = encode_level1_keys(tokens)
                opportunities[split] += doc_opportunities
                batch_events[split].extend(array for array in arrays if array.size)
                metadata_handle.write(
                    json.dumps(
                        {
                            "doc_id": documents_processed,
                            "relative_path": document.relative_path,
                            "line_number": document.line_number,
                            "split": "train" if is_train else "valid",
                            "num_tokens": int(tokens.size),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                documents_processed += 1

            if batch_events[0]:
                train_writer.write(np.concatenate(batch_events[0]))
            if batch_events[1]:
                valid_writer.write(np.concatenate(batch_events[1]))
            progress.update(
                documents_processed,
                int(token_totals.sum()),
                f"C-4-2 events={train_writer.events_written + valid_writer.events_written:,}",
            )
            if len(pending) < args.tokenizer_batch_size:
                break

    train_writer.close()
    valid_writer.close()
    np.save(cache_dir / "offsets.npy", np.asarray(offsets, dtype=np.uint64))
    np.save(cache_dir / "splits.npy", np.asarray(splits, dtype=np.uint8))
    np.save(cache_dir / "unigram_counts.npy", unigram_counts)
    np.save(cache_dir / "level1_opportunities.npy", opportunities)
    progress.update(
        documents_processed,
        int(token_totals.sum()),
        f"C-4-2 events={train_writer.events_written + valid_writer.events_written:,}",
        force=True,
    )
    return {
        "documents": documents_processed,
        "token_totals": token_totals,
        "unigram_counts": unigram_counts,
        "level1_opportunities": opportunities,
        "level1_events": np.asarray(
            [train_writer.events_written, valid_writer.events_written], dtype=np.uint64
        ),
        "max_length_documents": max_length_documents,
    }


def load_token_cache(cache_dir: Path):
    tokens = np.memmap(cache_dir / "tokens.bin", dtype=np.uint32, mode="r")
    offsets = np.load(cache_dir / "offsets.npy")
    splits = np.load(cache_dir / "splits.npy")
    metadata = []
    with (cache_dir / "documents.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            metadata.append(json.loads(line))
    return tokens, offsets, splits, metadata


def spool_level2(
    args: argparse.Namespace,
    output_dir: Path,
    active_keys_sorted: np.ndarray,
    active_ids_sorted: np.ndarray,
    active_count: int,
    logger: RunLogger,
) -> dict:
    cache_dir = output_dir / "cache"
    tokens, offsets, splits, _ = load_token_cache(cache_dir)
    id_bits = max(1, (active_count - 1).bit_length())
    event_root = output_dir / "_events" / "c8"
    writers = {
        ("real", 0): BucketWriter(event_root / "real" / "train", args.num_buckets),
        ("real", 1): BucketWriter(event_root / "real" / "valid", args.num_buckets),
        ("null", 0): BucketWriter(event_root / "null" / "train", args.num_buckets),
        ("null", 1): BucketWriter(event_root / "null" / "valid", args.num_buckets),
    }
    left_counts = {
        "real": np.zeros((2, LEVEL2_GAP_BUCKETS, active_count), dtype=np.uint64),
        "null": np.zeros((2, LEVEL2_GAP_BUCKETS, active_count), dtype=np.uint64),
    }
    right_counts = {
        "real": np.zeros((2, LEVEL2_GAP_BUCKETS, active_count), dtype=np.uint64),
        "null": np.zeros((2, LEVEL2_GAP_BUCKETS, active_count), dtype=np.uint64),
    }
    opportunities = {
        "real": np.zeros((2, LEVEL2_GAP_BUCKETS), dtype=np.uint64),
        "null": np.zeros((2, LEVEL2_GAP_BUCKETS), dtype=np.uint64),
    }
    pending: Dict[Tuple[str, int], List[np.ndarray]] = defaultdict(list)
    progress = ProgressReporter(
        logger, "C-8 real+null spool", len(splits), args.progress_every
    )
    cumulative_tokens = 0
    real_events = 0
    null_events = 0

    def flush_pending() -> None:
        for key, arrays in pending.items():
            if arrays:
                writers[key].write(np.concatenate(arrays))
        pending.clear()

    for doc_id, split_value in enumerate(splits):
        start = int(offsets[doc_id])
        end = int(offsets[doc_id + 1])
        doc_tokens = np.asarray(tokens[start:end], dtype=np.uint32)
        split = int(split_value)
        occurrences = active_occurrences(doc_tokens, active_keys_sorted, active_ids_sorted)
        rng = np.random.default_rng(args.seed + 1_000_003 * (doc_id + 1))
        null_occurrences = permute_occurrence_identities(occurrences, rng)

        for name, current_occurrences in (("real", occurrences), ("null", null_occurrences)):
            keys, left_ids, right_ids, gaps, _, _ = compose_occurrences(
                current_occurrences, len(doc_tokens), id_bits
            )
            if keys.size:
                pending[(name, split)].append(keys)
                np.add.at(left_counts[name][split], (gaps, left_ids), 1)
                np.add.at(right_counts[name][split], (gaps, right_ids), 1)
                opportunities[name][split] += np.bincount(
                    gaps, minlength=LEVEL2_GAP_BUCKETS
                ).astype(np.uint64)
            if name == "real":
                real_events += int(keys.size)
            else:
                null_events += int(keys.size)

        cumulative_tokens += len(doc_tokens)
        documents_done = doc_id + 1
        if documents_done % args.event_flush_documents == 0:
            flush_pending()
        progress.update(
            documents_done,
            cumulative_tokens,
            f"real events={real_events:,}, null events={null_events:,}",
        )

    flush_pending()
    for writer in writers.values():
        writer.close()
    progress.update(
        len(splits),
        cumulative_tokens,
        f"real events={real_events:,}, null events={null_events:,}",
        force=True,
    )
    for name in ("real", "null"):
        np.save(cache_dir / f"level2_{name}_left_counts.npy", left_counts[name])
        np.save(cache_dir / f"level2_{name}_right_counts.npy", right_counts[name])
        np.save(cache_dir / f"level2_{name}_opportunities.npy", opportunities[name])
    return {
        "id_bits": id_bits,
        "left_counts": left_counts,
        "right_counts": right_counts,
        "opportunities": opportunities,
        "event_counts": {"real": real_events, "null": null_events},
    }


def mine_level2(
    event_root: Path,
    name: str,
    num_buckets: int,
    id_bits: int,
    left_counts: np.ndarray,
    right_counts: np.ndarray,
    opportunities: np.ndarray,
    token_totals: np.ndarray,
    min_train_support: int,
    min_valid_support: int,
    min_npmi: float,
    max_log_rate_diff: float,
    beam_size: int,
    logger: RunLogger,
) -> Tuple[List[Candidate], dict]:
    heaps: Dict[Tuple[str, int], list] = defaultdict(list)
    stats = {
        "unique_train_candidates": 0,
        "pass_train_support": 0,
        "pass_valid_support": 0,
        "pass_npmi": 0,
        "pass_stability": 0,
        "pass_stability_hetero": 0,
        "pass_stability_self": 0,
    }
    train_tokens = max(int(token_totals[0]), 1)
    valid_tokens = max(int(token_totals[1]), 1)

    for bucket in range(num_buckets):
        train_keys, train_support = load_bucket_counts(
            event_root / name / "train" / f"bucket_{bucket:04d}.bin"
        )
        valid_keys, valid_support = load_bucket_counts(
            event_root / name / "valid" / f"bucket_{bucket:04d}.bin"
        )
        stats["unique_train_candidates"] += int(train_keys.size)
        mask = train_support >= min_train_support
        stats["pass_train_support"] += int(mask.sum())
        train_keys = train_keys[mask]
        train_support = train_support[mask]
        if train_keys.size == 0:
            continue
        matched_valid = lookup_counts(train_keys, valid_keys, valid_support)
        mask = matched_valid >= min_valid_support
        stats["pass_valid_support"] += int(mask.sum())
        train_keys = train_keys[mask]
        train_support = train_support[mask]
        matched_valid = matched_valid[mask]
        if train_keys.size == 0:
            continue

        gaps = (train_keys & 0b111).astype(np.int64)
        rights = ((train_keys >> 3) & ((1 << id_bits) - 1)).astype(np.int64)
        lefts = (train_keys >> (id_bits + 3)).astype(np.int64)
        totals = np.maximum(opportunities[0, gaps], 1)
        joint = train_support / totals
        left_prob = left_counts[0, gaps, lefts] / totals
        right_prob = right_counts[0, gaps, rights] / totals
        ratio = joint / np.maximum(left_prob * right_prob, 1e-300)
        denominator = np.maximum(-np.log(np.minimum(joint, 1 - 1e-15)), 1e-15)
        npmi_values = np.log(np.maximum(ratio, 1e-300)) / denominator
        mask = npmi_values >= min_npmi
        stats["pass_npmi"] += int(mask.sum())
        train_keys = train_keys[mask]
        train_support = train_support[mask]
        matched_valid = matched_valid[mask]
        gaps = gaps[mask]
        rights = rights[mask]
        lefts = lefts[mask]
        npmi_values = npmi_values[mask]
        if train_keys.size == 0:
            continue

        train_rates = train_support / train_tokens
        valid_rates = matched_valid / valid_tokens
        rate_diff = np.abs(np.log(np.maximum(valid_rates, 1e-300) / np.maximum(train_rates, 1e-300)))
        stable = rate_diff <= max_log_rate_diff
        stats["pass_stability"] += int(stable.sum())
        for key, left, right, gap, train_count, valid_count, score, stability in zip(
            train_keys[stable],
            lefts[stable],
            rights[stable],
            gaps[stable],
            train_support[stable],
            matched_valid[stable],
            npmi_values[stable],
            rate_diff[stable],
        ):
            kind = "self" if int(left) == int(right) else "hetero"
            stats[f"pass_stability_{kind}"] += 1
            candidate = Candidate(
                key=int(key),
                left=int(left),
                right=int(right),
                gap=int(gap),
                train_support=int(train_count),
                valid_support=int(valid_count),
                npmi=float(score),
                heldout_log_rate_diff=float(stability),
                kind=kind,
            )
            push_beam(heaps, candidate, beam_size)
        if (bucket + 1) % max(1, num_buckets // 8) == 0:
            logger.log(f"C-8 {name} counting: processed {bucket + 1}/{num_buckets} buckets")

    retained = flatten_beams(heaps)
    stats["retained_after_beam"] = len(retained)
    stats["retained_hetero"] = sum(item.kind == "hetero" for item in retained)
    stats["retained_self"] = sum(item.kind == "self" for item in retained)
    return retained, stats


def token_text(tokenizer, token_id: int) -> str:
    return tokenizer.decode([token_id], clean_up_tokenization_spaces=False)


def write_level1_csv(
    path: Path,
    candidates: Sequence[Candidate],
    tokenizer,
    total_documents: int,
) -> None:
    fields = [
        "pattern_id",
        "key",
        "left_token_id",
        "right_token_id",
        "distance",
        "left_text",
        "right_text",
        "train_support",
        "valid_support",
        "npmi",
        "heldout_log_rate_diff",
        "document_count",
        "document_coverage",
        "parent_reuse",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for pattern_id, item in enumerate(candidates):
            writer.writerow(
                {
                    "pattern_id": pattern_id,
                    "key": item.key,
                    "left_token_id": item.left,
                    "right_token_id": item.right,
                    "distance": item.gap + 1,
                    "left_text": token_text(tokenizer, item.left),
                    "right_text": token_text(tokenizer, item.right),
                    "train_support": item.train_support,
                    "valid_support": item.valid_support,
                    "npmi": f"{item.npmi:.8f}",
                    "heldout_log_rate_diff": f"{item.heldout_log_rate_diff:.8f}",
                    "document_count": item.document_count,
                    "document_coverage": f"{item.document_count / max(total_documents, 1):.8f}",
                    "parent_reuse": item.parent_reuse,
                }
            )


def describe_level1(candidate: Candidate, tokenizer) -> str:
    left = token_text(tokenizer, candidate.left)
    right = token_text(tokenizer, candidate.right)
    return f"{left!r} --d={candidate.gap + 1}--> {right!r}"


def write_level2_csv(
    path: Path,
    candidates: Sequence[Candidate],
    level1: Sequence[Candidate],
    tokenizer,
    total_documents: int,
) -> None:
    fields = [
        "key",
        "kind",
        "left_pattern_id",
        "right_pattern_id",
        "gap_bucket",
        "left_pattern",
        "right_pattern",
        "train_support",
        "valid_support",
        "npmi",
        "heldout_log_rate_diff",
        "document_count",
        "document_coverage",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in candidates:
            writer.writerow(
                {
                    "key": item.key,
                    "kind": item.kind,
                    "left_pattern_id": item.left,
                    "right_pattern_id": item.right,
                    "gap_bucket": item.gap,
                    "left_pattern": describe_level1(level1[item.left], tokenizer),
                    "right_pattern": describe_level1(level1[item.right], tokenizer),
                    "train_support": item.train_support,
                    "valid_support": item.valid_support,
                    "npmi": f"{item.npmi:.8f}",
                    "heldout_log_rate_diff": f"{item.heldout_log_rate_diff:.8f}",
                    "document_count": item.document_count,
                    "document_coverage": f"{item.document_count / max(total_documents, 1):.8f}",
                }
            )


def collect_coverage_and_examples(
    args: argparse.Namespace,
    output_dir: Path,
    tokenizer,
    level1: List[Candidate],
    level2_real: List[Candidate],
    level2_null: List[Candidate],
    active_keys_sorted: np.ndarray,
    active_ids_sorted: np.ndarray,
    id_bits: int,
    logger: RunLogger,
) -> List[dict]:
    tokens, offsets, splits, metadata = load_token_cache(output_dir / "cache")
    level1_doc_counts = np.zeros(len(level1), dtype=np.int64)
    level2_doc_counts = {
        "real": np.zeros(len(level2_real), dtype=np.int64),
        "null": np.zeros(len(level2_null), dtype=np.int64),
    }
    selected = {}
    for name, candidates in (("real", level2_real), ("null", level2_null)):
        keys = np.asarray([item.key for item in candidates], dtype=np.uint64)
        order = np.argsort(keys)
        selected[name] = (keys[order], order)
    top_keys = {
        item.key
        for item in sorted(level2_real, key=lambda value: -value.train_support)[
            : args.example_patterns
        ]
    }
    examples_by_key: Dict[int, List[dict]] = defaultdict(list)
    progress = ProgressReporter(logger, "coverage+examples", len(splits), args.progress_every)
    cumulative_tokens = 0

    for doc_id in range(len(splits)):
        start = int(offsets[doc_id])
        end = int(offsets[doc_id + 1])
        doc_tokens = np.asarray(tokens[start:end], dtype=np.uint32)
        occurrences = active_occurrences(doc_tokens, active_keys_sorted, active_ids_sorted)
        if occurrences:
            present_level1 = np.unique(
                np.fromiter((item.pattern_id for item in occurrences), dtype=np.int64)
            )
            level1_doc_counts[present_level1] += 1
        rng = np.random.default_rng(args.seed + 1_000_003 * (doc_id + 1))
        null_occurrences = permute_occurrence_identities(occurrences, rng)
        for name, current_occurrences, candidates in (
            ("real", occurrences, level2_real),
            ("null", null_occurrences, level2_null),
        ):
            if not candidates:
                continue
            keys, _, _, _, parent_starts, parent_ends = compose_occurrences(
                current_occurrences, len(doc_tokens), id_bits
            )
            if keys.size:
                unique_keys, first_indices = np.unique(keys, return_index=True)
                selected_keys_sorted, selected_order = selected[name]
                positions = np.searchsorted(selected_keys_sorted, unique_keys)
                in_bounds = positions < selected_keys_sorted.size
                safe = np.minimum(positions, max(selected_keys_sorted.size - 1, 0))
                matched = in_bounds & (selected_keys_sorted[safe] == unique_keys)
                candidate_indices = selected_order[positions[matched]]
                level2_doc_counts[name][candidate_indices] += 1
                if name != "real":
                    continue
                for key, event_index in zip(unique_keys[matched], first_indices[matched]):
                    key_int = int(key)
                    if key_int not in top_keys:
                        continue
                    if len(examples_by_key[key_int]) >= args.examples_per_pattern:
                        continue
                    parent_start = int(parent_starts[event_index])
                    parent_end = int(parent_ends[event_index])
                    context_start = max(0, parent_start - 8)
                    context_end = min(len(doc_tokens), parent_end + 9)
                    examples_by_key[key_int].append(
                        {
                            "key": key_int,
                            "relative_path": metadata[doc_id]["relative_path"],
                            "line_number": metadata[doc_id]["line_number"],
                            "split": metadata[doc_id]["split"],
                            "parent_start": parent_start,
                            "parent_end": parent_end,
                            "context": tokenizer.decode(
                                doc_tokens[context_start:context_end].tolist(),
                                clean_up_tokenization_spaces=False,
                            ),
                        }
                    )
        cumulative_tokens += len(doc_tokens)
        progress.update(doc_id + 1, cumulative_tokens)

    for pattern_id, count in enumerate(level1_doc_counts):
        level1[pattern_id].document_count = int(count)
    for candidate_index, count in enumerate(level2_doc_counts["real"]):
        level2_real[candidate_index].document_count = int(count)
    for candidate_index, count in enumerate(level2_doc_counts["null"]):
        level2_null[candidate_index].document_count = int(count)
    progress.update(len(splits), cumulative_tokens, force=True)
    return [example for examples in examples_by_key.values() for example in examples]


def compute_parent_reuse(level1: List[Candidate], level2: Sequence[Candidate]) -> dict:
    parent_sets: List[set] = [set() for _ in level1]
    for parent in level2:
        if parent.kind != "hetero":
            continue
        parent_sets[parent.left].add(parent.key)
        parent_sets[parent.right].add(parent.key)
    values = np.asarray([len(parents) for parents in parent_sets], dtype=np.int64)
    for index, value in enumerate(values):
        level1[index].parent_reuse = int(value)
    if values.size == 0:
        return {"patterns": 0}
    return {
        "patterns": int(values.size),
        "reused_by_at_least_1": int((values >= 1).sum()),
        "reused_by_at_least_2": int((values >= 2).sum()),
        "reused_by_at_least_5": int((values >= 5).sum()),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p99": float(np.percentile(values, 99)),
        "max": int(values.max()),
    }


def write_json(path: Path, value) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def run(args: argparse.Namespace) -> Path:
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}. Use a new path or --overwrite."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    logger = RunLogger(output_dir / "run.log")
    start_time = time.perf_counter()
    logger.log(f"output_dir={output_dir}")
    logger.log(f"loading tokenizer from {Path(args.tokenizer_dir).resolve()}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_dir, local_files_only=True, use_fast=True
    )
    if len(tokenizer) >= (1 << TOKEN_BITS):
        raise ValueError(f"Tokenizer has {len(tokenizer)} ids, exceeding {TOKEN_BITS}-bit key encoding")

    config = vars(args).copy()
    config["data_dir"] = str(Path(args.data_dir).resolve())
    config["tokenizer_dir"] = str(Path(args.tokenizer_dir).resolve())
    config["output_dir"] = str(output_dir)
    config["tokenizer_class"] = type(tokenizer).__name__
    config["tokenizer_size"] = len(tokenizer)
    write_json(output_dir / "run_config.json", config)

    logger.log("stage 1/5: tokenize documents and spool exact C-4-2 events")
    corpus = tokenize_and_spool_level1(args, tokenizer, output_dir, logger)
    logger.log("stage 2/5: count and select C-4-2 patterns")
    level1, level1_stats = mine_level1(
        output_dir / "_events" / "c4",
        args.num_buckets,
        corpus["unigram_counts"],
        corpus["token_totals"],
        corpus["level1_opportunities"],
        args.level1_min_train_support,
        args.level1_min_valid_support,
        args.level1_min_npmi,
        args.max_log_rate_diff,
        args.beam_per_support_bucket,
        logger,
    )
    logger.log(f"retained C-4-2 patterns: {len(level1):,}")
    write_level1_csv(
        output_dir / "c4_patterns_preliminary.csv",
        level1,
        tokenizer,
        corpus["documents"],
    )

    summary = {
        "status": "incomplete",
        "corpus": {
            "documents": corpus["documents"],
            "train_tokens": int(corpus["token_totals"][0]),
            "valid_tokens": int(corpus["token_totals"][1]),
            "total_tokens": int(corpus["token_totals"].sum()),
            "documents_at_max_length": corpus["max_length_documents"],
            "level1_events_train": int(corpus["level1_events"][0]),
            "level1_events_valid": int(corpus["level1_events"][1]),
        },
        "c4": level1_stats,
    }
    write_json(output_dir / "summary.json", summary)
    if not level1:
        summary["status"] = "insufficient_evidence_no_level1_patterns"
        summary["elapsed_seconds"] = time.perf_counter() - start_time
        write_json(output_dir / "summary.json", summary)
        logger.log("no C-4-2 patterns survived; stopping before C-8")
        return output_dir

    active_keys = np.asarray([item.key for item in level1], dtype=np.uint64)
    active_ids = np.arange(len(level1), dtype=np.int64)
    active_order = np.argsort(active_keys)
    active_keys_sorted = active_keys[active_order]
    active_ids_sorted = active_ids[active_order]

    logger.log("stage 3/5: spool real and occurrence-permuted C-8 events")
    level2_spool = spool_level2(
        args,
        output_dir,
        active_keys_sorted,
        active_ids_sorted,
        len(level1),
        logger,
    )
    logger.log("stage 4/5: count and select real/null C-8 patterns")
    level2_real, level2_real_stats = mine_level2(
        output_dir / "_events" / "c8",
        "real",
        args.num_buckets,
        level2_spool["id_bits"],
        level2_spool["left_counts"]["real"],
        level2_spool["right_counts"]["real"],
        level2_spool["opportunities"]["real"],
        corpus["token_totals"],
        args.level2_min_train_support,
        args.level2_min_valid_support,
        args.level2_min_npmi,
        args.max_log_rate_diff,
        args.beam_per_support_bucket,
        logger,
    )
    level2_null, level2_null_stats = mine_level2(
        output_dir / "_events" / "c8",
        "null",
        args.num_buckets,
        level2_spool["id_bits"],
        level2_spool["left_counts"]["null"],
        level2_spool["right_counts"]["null"],
        level2_spool["opportunities"]["null"],
        corpus["token_totals"],
        args.level2_min_train_support,
        args.level2_min_valid_support,
        args.level2_min_npmi,
        args.max_log_rate_diff,
        args.beam_per_support_bucket,
        logger,
    )
    logger.log(
        f"retained C-8: real={len(level2_real):,}, null={len(level2_null):,}"
    )

    logger.log("stage 5/5: document coverage, examples, and parent reuse")
    examples = collect_coverage_and_examples(
        args,
        output_dir,
        tokenizer,
        level1,
        level2_real,
        level2_null,
        active_keys_sorted,
        active_ids_sorted,
        level2_spool["id_bits"],
        logger,
    )
    parent_reuse = compute_parent_reuse(level1, level2_real)
    write_level1_csv(
        output_dir / "c4_patterns.csv", level1, tokenizer, corpus["documents"]
    )
    write_level2_csv(
        output_dir / "c8_real_patterns.csv",
        level2_real,
        level1,
        tokenizer,
        corpus["documents"],
    )
    write_level2_csv(
        output_dir / "c8_null_patterns.csv",
        level2_null,
        level1,
        tokenizer,
        corpus["documents"],
    )
    with (output_dir / "examples.jsonl").open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example, ensure_ascii=False) + "\n")

    real_pass = level2_real_stats["pass_stability_hetero"]
    null_pass = level2_null_stats["pass_stability_hetero"]
    summary.update(
        {
            "status": "complete",
            "c8_real": level2_real_stats,
            "c8_null": level2_null_stats,
            "c8_events": level2_spool["event_counts"],
            "hetero_pass_count_ratio_real_over_null": (
                real_pass / null_pass if null_pass > 0 else None
            ),
            "parent_reuse": parent_reuse,
            "examples_saved": len(examples),
            "elapsed_seconds": time.perf_counter() - start_time,
        }
    )
    write_json(output_dir / "summary.json", summary)
    if not args.keep_event_files:
        shutil.rmtree(output_dir / "_events", ignore_errors=True)
    logger.log(f"complete in {format_duration(summary['elapsed_seconds'])}")
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Mine hierarchical token co-occurrence patterns from DCLM text."
    )
    parser.add_argument("--data-dir", default="/Users/bytedance/Desktop/dclm")
    parser.add_argument("--tokenizer-dir", default="fdong/Qwen3-0.6B")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-documents", type=int, default=10_000)
    parser.add_argument("--max-documents-per-file", type=int, default=128)
    parser.add_argument("--max-tokens-per-document", type=int, default=1_024)
    parser.add_argument("--tokenizer-batch-size", type=int, default=16)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--num-buckets", type=int, default=128)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--event-flush-documents", type=int, default=32)
    parser.add_argument("--level1-min-train-support", type=int, default=100)
    parser.add_argument("--level1-min-valid-support", type=int, default=20)
    parser.add_argument("--level1-min-npmi", type=float, default=0.10)
    parser.add_argument("--level2-min-train-support", type=int, default=20)
    parser.add_argument("--level2-min-valid-support", type=int, default=5)
    parser.add_argument("--level2-min-npmi", type=float, default=0.10)
    parser.add_argument("--max-log-rate-diff", type=float, default=math.log(2.0))
    parser.add_argument("--beam-per-support-bucket", type=int, default=10_000)
    parser.add_argument("--example-patterns", type=int, default=100)
    parser.add_argument("--examples-per-pattern", type=int, default=3)
    parser.add_argument("--keep-event-files", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not 0 < args.train_fraction < 1:
        raise ValueError("--train-fraction must be between 0 and 1")
    if args.num_buckets <= 0:
        raise ValueError("--num-buckets must be positive")
    if args.max_tokens_per_document < LEVEL2_MAX_SPAN:
        raise ValueError("--max-tokens-per-document must be at least 8")
    run(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

import sys
import unittest
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mine_hierarchy import (  # noqa: E402
    Occurrence,
    active_occurrences,
    compose_occurrences,
    decode_level1_key,
    decode_level2_key,
    encode_level1_keys,
    encode_level2_key,
    permute_occurrence_identities,
)


class Level1Tests(unittest.TestCase):
    def test_all_pairs_with_span_at_most_four(self):
        tokens = np.asarray([10, 20, 30, 40], dtype=np.uint32)
        arrays, opportunities = encode_level1_keys(tokens)
        self.assertEqual(opportunities.tolist(), [3, 2, 1])
        decoded = [decode_level1_key(int(key)) for array in arrays for key in array]
        self.assertEqual(
            decoded,
            [
                (10, 20, 0),
                (20, 30, 0),
                (30, 40, 0),
                (10, 30, 1),
                (20, 40, 1),
                (10, 40, 2),
            ],
        )

    def test_active_occurrences_reconstruct_positions(self):
        tokens = np.asarray([10, 20, 30, 40], dtype=np.uint32)
        arrays, _ = encode_level1_keys(tokens)
        active_keys = np.asarray([arrays[0][0], arrays[0][2]], dtype=np.uint64)
        order = np.argsort(active_keys)
        occurrences = active_occurrences(
            tokens,
            active_keys[order],
            np.asarray([0, 1], dtype=np.int64)[order],
        )
        self.assertEqual(
            occurrences,
            [Occurrence(0, 0, 1), Occurrence(1, 2, 3)],
        )


class Level2Tests(unittest.TestCase):
    def test_nonoverlapping_pair_of_pairs(self):
        occurrences = [
            Occurrence(0, 0, 1),
            Occurrence(1, 1, 2),
            Occurrence(2, 2, 3),
            Occurrence(3, 4, 5),
        ]
        id_bits = 3
        keys, lefts, rights, gaps, starts, ends = compose_occurrences(
            occurrences, document_length=6, id_bits=id_bits
        )
        decoded = [decode_level2_key(int(key), id_bits) for key in keys]
        self.assertIn((0, 2, 0), decoded)
        self.assertIn((0, 3, 2), decoded)
        self.assertNotIn((0, 1, 0), decoded)
        self.assertTrue(np.all(ends - starts + 1 <= 8))
        self.assertEqual(len(keys), len(lefts))
        self.assertEqual(len(keys), len(rights))
        self.assertEqual(len(keys), len(gaps))

    def test_level2_key_round_trip(self):
        key = encode_level2_key(123, 456, 2, id_bits=10)
        self.assertEqual(decode_level2_key(key, id_bits=10), (123, 456, 2))

    def test_permutation_preserves_pattern_counts_by_span(self):
        occurrences = [
            Occurrence(0, 0, 1),
            Occurrence(1, 2, 3),
            Occurrence(0, 4, 6),
            Occurrence(2, 7, 9),
        ]
        permuted = permute_occurrence_identities(
            occurrences, np.random.default_rng(7)
        )

        def grouped_counts(values):
            result = defaultdict(Counter)
            for value in values:
                result[value.end - value.start + 1][value.pattern_id] += 1
            return result

        self.assertEqual(grouped_counts(occurrences), grouped_counts(permuted))
        self.assertEqual(
            sorted((item.start, item.end) for item in occurrences),
            sorted((item.start, item.end) for item in permuted),
        )


if __name__ == "__main__":
    unittest.main()

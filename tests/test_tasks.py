import unittest
from decimal import Decimal

from q4tasks import bootstrap_diff, clustered_mgsm_diff, parse_code, parse_number


class TestTaskEvaluation(unittest.TestCase):
    def test_number_strict(self):
        self.assertEqual(parse_number("Compute 2 + 5.\nFINAL: 7"), Decimal(7))
        self.assertEqual(parse_number("FINAL: -1,234"), Decimal(-1234))
        self.assertIsNone(parse_number("The result is 7"))
        self.assertIsNone(parse_number("FINAL: 7 dollars"))
        self.assertEqual(parse_number("FINAL: 1\nFINAL: 2"), Decimal(2))

    def test_extract_code(self):
        fence = chr(96) * 3
        self.assertEqual(parse_code(fence + "python\n" + "def foo():\n    return 1\n" + fence), "def foo():\n    return 1")
        self.assertEqual(parse_code("def foo():\n    return 1"), "def foo():\n    return 1")

    def test_bilingual_cluster(self):
        a = {"mgsm-en-001": {"correct": True}, "mgsm-zh-001": {"correct": True}, "mgsm-en-002": {"correct": False}, "mgsm-zh-002": {"correct": False}}
        b = {"mgsm-en-001": {"correct": False}, "mgsm-zh-001": {"correct": True}, "mgsm-en-002": {"correct": False}, "mgsm-zh-002": {"correct": False}}
        result = clustered_mgsm_diff(a, b, iterations=400)
        self.assertEqual(result["question_pairs"], 2)
        self.assertEqual(result["language_responses"], 4)
        self.assertAlmostEqual(result["accuracy_delta"], 0.25)

    def test_paired_bootstrap(self):
        a = {"q1": {"correct": True}, "q2": {"correct": False}, "q3": {"correct": True}}
        b = {"q1": {"correct": False}, "q2": {"correct": True}, "q3": {"correct": True}}
        result = bootstrap_diff(a, b, iterations=400)
        self.assertEqual(result["n"], 3)
        self.assertEqual(result["first_only"], 1)
        self.assertEqual(result["second_only"], 1)
        self.assertEqual(result["both_correct"], 1)
        self.assertAlmostEqual(result["delta"], 0)


if __name__ == "__main__":
    unittest.main()

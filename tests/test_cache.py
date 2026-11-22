import json
import struct
import tempfile
import unittest
from pathlib import Path

from gguf import GGMLQuantizationType

from q4cache import align_up, compose, load_index, prepare
from q4fit import BLOCKS, digest


def make_gguf(path, types, file_type):
    names = ["blk.0.attn_q.weight", "blk.0.ffn_down.weight", "output_norm.weight"]
    shapes = [[256, 1], [256, 1], [256]]
    header = bytearray(b"GGUF" + struct.pack("<IQQ", 3, len(names), 2))
    for key, value in [("general.alignment", 32), ("general.file_type", file_type)]:
        key = key.encode()
        header += struct.pack("<Q", len(key)) + key + struct.pack("<II", 4, value)
    data = []
    pos = 0
    for i, (name, shape, typ) in enumerate(zip(names, shapes, types)):
        kind = GGMLQuantizationType[typ]
        b = name.encode()
        count = 256 if len(shape) == 1 else 256
        if typ == "F32":
            size = count * 4
        else:
            block, width = BLOCKS[typ]
            size = count // block * width
        pos = align_up(pos, 32)
        header += struct.pack("<Q", len(b)) + b + struct.pack("<I", len(shape))
        header += struct.pack("<" + "Q" * len(shape), *shape)
        header += struct.pack("<IQ", int(kind), pos)
        data.append((pos, bytes([(int(kind) + i) % 251]) * size))
        pos += size
    header += bytes(align_up(len(header), 32) - len(header))
    with open(path, "wb") as output:
        output.write(header)
        at = 0
        for offset, payload in data:
            output.write(bytes(offset - at))
            output.write(payload)
            at = offset + len(payload)
        output.write(bytes(align_up(at, 32) - at))


class TestQ4Cache(unittest.TestCase):
    def test_exact_compose(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)
            src = p / "src.gguf"
            make_gguf(src, ["BF16", "BF16", "F32"], 0)
            shared = p / "banks"
            shared.mkdir()
            files = {}
            for typ in ("Q3_K", "Q4_K", "Q5_K"):
                path = shared / (typ + ".gguf")
                make_gguf(path, [typ, typ, "F32"], 12 if typ == "Q4_K" else 11 if typ == "Q3_K" else 13)
                path.with_suffix(".manifest.json").write_text(json.dumps({
                    "source_sha256": digest(src), "default": typ, "overrides": {}, "imatrix": None
                }))
                files[typ] = path
            idx = prepare(src, shared, list(files), None, 1, files)
            self.assertEqual(set(load_index(idx)["banks"]), set(files))
            out = p / "mixed.gguf"
            got = compose(idx, out, changes=[
                r"^blk\.0\.attn_q\.weight$=Q3_K",
                r"^blk\.0\.ffn_down\.weight$=Q5_K",
            ])
            reference = p / "reference.gguf"
            make_gguf(reference, ["Q3_K", "Q5_K", "F32"], 12)
            self.assertEqual(out.read_bytes(), reference.read_bytes())
            self.assertEqual(got["output_sha256"], digest(reference))
            out.unlink()
            fast = compose(idx, out, changes=[
                r"^blk\.0\.attn_q\.weight$=Q3_K",
                r"^blk\.0\.ffn_down\.weight$=Q5_K",
            ], verify_output=False, hash_output=False)
            self.assertEqual(out.read_bytes(), reference.read_bytes())
            self.assertIsNone(fast["output_sha256"])
            with self.assertRaises(ValueError):
                compose(idx, files["Q4_K"])
            with self.assertRaises(ValueError):
                compose(idx, p / "bad.gguf", changes=[r"^blk\.0\.attn_q\.weight$=Q8_0"])
            files["Q3_K"].write_bytes(files["Q3_K"].read_bytes() + b"\x00")
            with self.assertRaises(ValueError):
                load_index(idx)

    def test_alignment(self):
        self.assertEqual(align_up(128, 32), 128)
        self.assertEqual(align_up(129, 32), 160)


if __name__ == "__main__":
    unittest.main()

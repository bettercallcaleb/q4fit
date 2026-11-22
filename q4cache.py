#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import shutil
import struct
import tempfile
import time
from pathlib import Path

from q4fit import BLOCKS, build, digest, inventory, override_map

INDEX_VERSION = 1
DEFAULT_ALIGNMENT = 32


def align_up(value, alignment):
    return (value + alignment - 1) // alignment * alignment


def read_gguf(path):
    from gguf import GGUFReader
    reader = GGUFReader(str(path), "r")
    if bytes(reader.data[:4]) != b"GGUF" or struct.unpack_from("<I", reader.data, 4)[0] != 3:
        raise ValueError("Only little-endian GGUF v3 is supported")
    if reader.data_offset < 32 or reader.data_offset > 256 * 1024 * 1024:
        raise ValueError("Unexpected GGUF header size")
    names = [t.name for t in reader.tensors]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate tensor names")
    return reader


def header_positions(reader):
    base_address = reader.data.ctypes.data
    offsets = {}
    for t in reader.tensors:
        parts = t.field.parts
        if len(parts) < 2 or parts[-2].dtype.kind != "u" or parts[-2].dtype.itemsize != 4 or parts[-1].dtype.kind != "u" or parts[-1].dtype.itemsize != 8:
            raise ValueError("Unsupported tensor descriptor")
        typ = parts[-2].ctypes.data - base_address
        off = parts[-1].ctypes.data - base_address
        if not (0 <= typ < reader.data_offset - 4 and 0 <= off < reader.data_offset - 8):
            raise ValueError("Descriptor outside header")
        offsets[t.name] = (int(typ), int(off))
    return offsets


def metadata_fingerprint(reader):
    h = hashlib.sha256()
    for key, field in reader.fields.items():
        if key == "general.file_type":
            continue
        h.update(key.encode("utf-8"))
        h.update(b"\x00")
        for part in field.parts:
            h.update(part.tobytes())
    return h.hexdigest()


def bank_info(path):
    from gguf import GGMLQuantizationType
    r = read_gguf(path)
    tensors = {}
    for t in r.tensors:
        tensors[t.name] = {
            "type": GGMLQuantizationType(t.tensor_type).name,
            "type_id": int(t.tensor_type),
            "shape": [int(n) for n in t.shape],
            "bytes": int(t.n_bytes),
            "offset": int(t.data_offset),
        }
    return {
        "metadata_sha256": metadata_fingerprint(r),
        "alignment": int(r.alignment),
        "data_offset": int(r.data_offset),
        "tensor_order": [t.name for t in r.tensors],
        "tensors": tensors,
    }


def parse_reuse(items):
    result = {}
    for entry in items:
        if "=" not in entry:
            raise ValueError("Expected TYPE=PATH for --reuse")
        typ, location = entry.split("=", 1)
        typ = typ.upper()
        if typ not in BLOCKS or typ in result:
            raise ValueError("Unknown or duplicate bank type")
        result[typ] = Path(location).resolve()
    return result


def prepare(source, cache, types, quantizer, threads, reuse=None, imatrix=None):
    source = Path(source).resolve()
    cache = Path(cache).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    if not source.is_file():
        raise FileNotFoundError(source)
    source_hash = digest(source)
    source_tensors = inventory(source)
    types = [t.upper() for t in types]
    if not types or len(types) != len(set(types)) or "Q4_K" not in types:
        raise ValueError("Types must be unique and include Q4_K")
    for t in types:
        if t not in BLOCKS:
            raise ValueError("Unsupported bank type: " + t)
    reuse = reuse or {}
    imatrix_hash = digest(imatrix) if imatrix else None
    all_banks = {}
    signature = None
    template_names = None
    for typ in types:
        path = reuse.get(typ, cache / (typ + ".gguf"))
        path = Path(path).resolve()
        if path == source:
            raise ValueError("A quantized bank cannot be the BF16 source")
        if not path.is_file():
            if typ in reuse:
                raise FileNotFoundError(path)
            if not quantizer:
                raise ValueError("Missing --quantizer for creating " + typ)
            print("PREQUANTIZE", typ, flush=True)
            build(source, path, quantizer, typ, [], str(imatrix) if imatrix else None, threads)
        manifest_file = path.with_suffix(".manifest.json")
        if not manifest_file.exists():
            raise ValueError("Missing Q4Fit provenance receipt: " + str(manifest_file))
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        if manifest.get("source_sha256") != source_hash or manifest.get("default") != typ or manifest.get("overrides") != {}:
            raise ValueError("Bank was not built from the same BF16 source and requested type: " + str(path))
        recorded_imatrix = manifest.get("imatrix")
        if (digest(recorded_imatrix) if recorded_imatrix else None) != imatrix_hash:
            raise ValueError("Incompatible imatrix for bank: " + typ)
        info = bank_info(path)
        if signature is None:
            signature = (info["metadata_sha256"], info["alignment"], info["tensor_order"])
            template_names = set(info["tensors"])
        elif (info["metadata_sha256"], info["alignment"], info["tensor_order"]) != signature:
            raise ValueError("Incompatible GGUF metadata/tensor order in " + typ)
        if set(info["tensors"]) != template_names:
            raise ValueError("Missing tensor from bank")
        for t in source_tensors:
            item = info["tensors"].get(t["name"])
            if item is None or item["shape"] != t["shape"]:
                raise ValueError("Bank tensor shape mismatch: " + t["name"])
        for entry in info["tensors"].values():
            if entry["offset"] + entry["bytes"] > path.stat().st_size:
                raise ValueError("Truncated tensor payload in " + typ)
        stat = path.stat()
        all_banks[typ] = {
            "path": str(path),
            "bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": digest(path),
            **info,
        }
        print("BANK", typ, "tensors", len(info["tensors"]), "GiB", round(stat.st_size / 2 ** 30, 3), flush=True)
    base_reader = read_gguf(all_banks["Q4_K"]["path"])
    base_header = bytes(base_reader.data[:base_reader.data_offset])
    index = {
        "version": INDEX_VERSION,
        "base_header_sha256": hashlib.sha256(base_header).hexdigest(),
        "base_data_offset": int(base_reader.data_offset),
        "base_descriptor_positions": header_positions(base_reader),
        "source": str(source),
        "source_sha256": source_hash,
        "source_tensors": source_tensors,
        "metadata_sha256": signature[0],
        "imatrix_sha256": imatrix_hash,
        "banks": all_banks,
        "default": "Q4_K",
    }
    index_file = cache / "index.json"
    tmp = cache / "index.json.tmp"
    tmp.write_text(json.dumps(index, indent=2), encoding="utf-8")
    os.replace(tmp, index_file)
    print("CACHE", index_file, flush=True)
    return index_file


def load_index(index_path, verify_sha=False):
    data = json.loads(Path(index_path).read_text(encoding="utf-8"))
    if data.get("version") != INDEX_VERSION:
        raise ValueError("Unknown cache version")
    if "Q4_K" not in data.get("banks", {}):
        raise ValueError("Missing base Q4_K bank")
    for typ, info in data["banks"].items():
        path = Path(info["path"])
        stat = path.stat()
        if stat.st_size != info["bytes"] or stat.st_mtime_ns != info["mtime_ns"]:
            raise ValueError("Cached bank changed since indexing: " + typ)
        if verify_sha and digest(path) != info["sha256"]:
            raise ValueError("Cached bank SHA256 mismatch: " + typ)
    return data


def send_bytes(dst, src, pos, count):
    remaining = count
    while remaining:
        n = os.sendfile(dst, src, pos, remaining)
        if n <= 0:
            raise OSError("Unexpected EOF copying tensor data")
        pos += n
        remaining -= n


def compose(index_path, output, default="Q4_K", changes=None, verify_banks=False, verify_output=True, hash_output=True):
    from gguf import GGMLQuantizationType
    start = time.perf_counter()
    index = load_index(index_path, verify_sha=verify_banks)
    banks = index["banks"]
    if default not in banks:
        raise ValueError("Missing default bank " + default)
    if default != "Q4_K":
        raise ValueError("Cache compose currently requires Q4_K as the default")
    changes = changes or []
    overrides = override_map(index["source_tensors"], changes)
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if any(output == Path(b["path"]).resolve() for b in banks.values()):
        raise ValueError("Output path must not overwrite a source bank")
    base = banks["Q4_K"]
    with open(base["path"], "rb") as fd:
        header = bytearray(fd.read(index["base_data_offset"]))
    if hashlib.sha256(header).hexdigest() != index["base_header_sha256"]:
        raise ValueError("Base header modified since cache indexing")
    positions = index["base_descriptor_positions"]
    order = base["tensor_order"]
    alignment = base["alignment"]
    layout = []
    cursor = 0
    for name in order:
        typ = overrides.get(name, default)
        chosen = banks.get(typ)
        if chosen is None:
            raise ValueError("Missing requested quant bank " + typ)
        item = chosen["tensors"][name]
        source = next((t for t in index["source_tensors"] if t["name"] == name), None)
        if source is None or source["shape"] != item["shape"]:
            raise ValueError("Tensor shape mismatch " + name)
        if typ != default and item["type"] != typ:
            raise ValueError("Bank " + typ + " did not quantize tensor " + name + " to the required type")
        if typ == default and item["type"] != base["tensors"][name]["type"]:
            raise ValueError("Invalid default bank")
        cursor = align_up(cursor, alignment)
        typ_offset, offset_offset = positions[name]
        struct.pack_into("<I", header, typ_offset, item["type_id"])
        struct.pack_into("<Q", header, offset_offset, cursor)
        layout.append((name, typ, item["offset"], item["bytes"], cursor))
        cursor += item["bytes"]
    for name, typ, src_pos, nbytes, dest_offset in layout:
        bank_info_entry = banks[typ]
        if src_pos + nbytes > bank_info_entry["bytes"]:
            raise ValueError("Bank range outside file: " + name)
    free = shutil.disk_usage(output.parent).free
    expected_size = base["data_offset"] + align_up(cursor, alignment)
    if free < expected_size + 2 ** 30:
        raise RuntimeError("Insufficient free space for destination model")
    source_fds = {typ: os.open(b["path"], os.O_RDONLY) for typ, b in banks.items()}
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w+b", dir=output.parent, prefix="." + output.name + ".", suffix=".part", delete=False, buffering=0) as tmp:
            tmp_path = Path(tmp.name)
            fd = tmp.fileno()
            os.write(fd, header)
            written_offset = 0
            for name, typ, pos, nbytes, offset in layout:
                if offset > written_offset:
                    os.write(fd, bytes(offset - written_offset))
                send_bytes(fd, source_fds[typ], pos, nbytes)
                written_offset = offset + nbytes
            pad = align_up(written_offset, alignment) - written_offset
            if pad:
                os.write(fd, bytes(pad))
            os.fsync(fd)
        if verify_output:
            fresh = read_gguf(tmp_path)
            if [t.name for t in fresh.tensors] != order:
                raise ValueError("Output tensor-order mismatch")
            if len(fresh.tensors) != len(layout):
                raise ValueError("Output tensor count mismatch")
            for t, (name, typ, pos, nbytes, rel_offset) in zip(fresh.tensors, layout):
                expected = banks[typ]["tensors"][name]
                if int(t.tensor_type) != expected["type_id"] or int(t.n_bytes) != nbytes or int(t.data_offset) != base["data_offset"] + rel_offset:
                    raise ValueError("Invalid generated GGUF tensor descriptor " + name)
        os.replace(tmp_path, output)
        tmp_path = None
        result = {
            "output": str(output),
            "output_sha256": digest(output) if hash_output else None,
            "bytes": output.stat().st_size,
            "index": str(Path(index_path).resolve()),
            "index_sha256": digest(index_path),
            "source_sha256": index["source_sha256"],
            "default": default,
            "overrides": overrides,
            "compose_seconds": round(time.perf_counter() - start, 3),
            "tensor_count": len(layout),
            "verified_gguf": verify_output,
        }
        output.with_suffix(".manifest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2), flush=True)
        return result
    finally:
        for fd in source_fds.values():
            os.close(fd)
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


def main():
    p = argparse.ArgumentParser(description="Q4Fit prequantized tensor-bank GGUF composer")
    sub = p.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--model", required=True)
    prep.add_argument("--cache", required=True)
    prep.add_argument("--types", default="Q3_K,Q4_K,Q5_K")
    prep.add_argument("--reuse", action="append", default=[], metavar="TYPE=GGUF")
    prep.add_argument("--quantizer", default="llama-quantize")
    prep.add_argument("--threads", type=int, default=12)
    prep.add_argument("--imatrix")
    comp = sub.add_parser("compose")
    comp.add_argument("--index", required=True)
    comp.add_argument("--output", required=True)
    comp.add_argument("--default", default="Q4_K")
    comp.add_argument("--set", action="append", default=[], metavar="REGEX=TYPE")
    comp.add_argument("--verify-banks", action="store_true")
    comp.add_argument("--fast", action="store_true")
    comp.add_argument("--no-verify-output", action="store_true")
    audit = sub.add_parser("audit")
    audit.add_argument("--index", required=True)
    audit.add_argument("--verify-banks", action="store_true")
    args = p.parse_args()
    if args.command == "prepare":
        prepare(args.model, args.cache, args.types.split(","), args.quantizer, args.threads, parse_reuse(args.reuse), args.imatrix)
    elif args.command == "compose":
        compose(args.index, args.output, args.default, args.set, args.verify_banks, not (args.no_verify_output or args.fast), not args.fast)
    else:
        data = load_index(args.index, args.verify_banks)
        print(json.dumps({"source_sha256": data["source_sha256"], "banks": {t: {"bytes": v["bytes"], "sha256": v["sha256"], "tensor_count": len(v["tensors"])} for t, v in data["banks"].items()}}, indent=2))


if __name__ == "__main__":
    main()

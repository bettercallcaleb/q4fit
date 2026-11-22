#!/usr/bin/env python3
import argparse
import hashlib
import json
import zipfile
from pathlib import Path

def main():
    p = argparse.ArgumentParser(description="Extract WikiText-2 raw validation/test from a user-supplied archive")
    p.add_argument("--archive", required=True)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    records = {}
    with zipfile.ZipFile(args.archive) as z:
        for split in ("valid", "test"):
            name = f"wikitext-2-raw/wiki.{split}.raw"
            content = z.read(name)
            path = output / f"wikitext2_{split}.raw.txt"
            path.write_bytes(content)
            records[split] = {
                "path": str(path),
                "sha256": hashlib.sha256(content).hexdigest(),
                "bytes": len(content),
                "source_archive": str(Path(args.archive).resolve()),
            }
    (output / "corpora.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(json.dumps(records, indent=2))

if __name__ == "__main__":
    main()

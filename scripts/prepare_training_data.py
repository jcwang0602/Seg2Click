"""Build the indexed JSONL training format (GUIAIDX1) without altering JSON sources.

Usage (from the repository root)::

    python scripts/prepare_training_data.py --check-only  # validate prepared data
    python scripts/prepare_training_data.py               # first-time preparation

The CLI always reads data/data_config.yaml. Existing JSONL/index files are never
replaced; use --check-only to validate an installation that is already prepared.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import tempfile
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_CONFIG = REPO_ROOT / "data/data_config.yaml"
# On-disk contract consumed by seg2click.dataset.IndexedJsonlSource.
INDEX_MAGIC = b"GUIAIDX1"
INDEX_HEADER = struct.Struct("<8sQQ")
OFFSET = struct.Struct("<Q")


def iter_json_records(path: Path, chunk_size: int = 1 << 20):
    """Read a JSON array incrementally, retaining at most one pending record."""
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as source:
        buffer, position, finished = "", 0, False

        def read_more():
            nonlocal buffer, position, finished
            chunk = source.read(chunk_size)
            buffer = buffer[position:] + chunk
            position = 0
            finished = not chunk

        def skip_space():
            nonlocal position
            while True:
                while position < len(buffer) and buffer[position] in " \t\r\n":
                    position += 1
                if position < len(buffer) or finished:
                    return
                read_more()

        skip_space()
        if position >= len(buffer) or buffer[position] != "[":
            raise ValueError(f"Expected a JSON array: {path}")
        position += 1
        skip_space()
        if position < len(buffer) and buffer[position] == "]":
            position += 1
        else:
            while True:
                skip_space()
                while True:
                    try:
                        record, end = decoder.raw_decode(buffer, position)
                        break
                    except json.JSONDecodeError:
                        if finished:
                            raise
                        read_more()
                if not isinstance(record, dict):
                    raise ValueError(f"Training records must be JSON objects: {path}")
                position = end
                yield record
                skip_space()
                if position >= len(buffer):
                    raise ValueError(f"Unterminated JSON array: {path}")
                delimiter = buffer[position]
                position += 1
                if delimiter == "]":
                    break
                if delimiter != ",":
                    raise ValueError(f"Expected ',' or ']' after a record: {path}")
                skip_space()
                if position < len(buffer) and buffer[position] == "]":
                    raise ValueError(f"Trailing comma in JSON array: {path}")
        skip_space()
        if position < len(buffer):
            raise ValueError(f"Unexpected content after JSON array: {path}")


def check_pair(jsonl_path: Path, index_path: Path) -> dict:
    """Check the complete offset table plus first/last indexed JSON objects.

    This checks the format and file sizes, not every image or source annotation.
    It deliberately does not open the original multi-gigabyte JSON arrays.
    """
    byte_count = jsonl_path.stat().st_size
    with index_path.open("rb") as index:
        header = index.read(INDEX_HEADER.size)
        if len(header) != INDEX_HEADER.size:
            raise ValueError(f"Truncated GUIAIDX1 header: {index_path}")
        magic, record_count, recorded_bytes = INDEX_HEADER.unpack(header)
        if magic != INDEX_MAGIC or recorded_bytes != byte_count:
            raise ValueError(f"Invalid GUIAIDX1 magic or JSONL byte count: {index_path}")
        if index_path.stat().st_size != INDEX_HEADER.size + record_count * OFFSET.size:
            raise ValueError(f"Invalid GUIAIDX1 offset-table size: {index_path}")
        first, previous = None, -1
        remaining = record_count
        while remaining:
            count = min(remaining, 8192)
            block = index.read(count * OFFSET.size)
            if len(block) != count * OFFSET.size:
                raise ValueError(f"Truncated GUIAIDX1 offset table: {index_path}")
            for (offset,) in struct.iter_unpack("<Q", block):
                if not previous < offset < byte_count:
                    raise ValueError(f"Invalid/non-increasing JSONL offset: {index_path}")
                if first is None:
                    first = offset
                previous = offset
            remaining -= count
    if record_count == 0:
        if byte_count:
            raise ValueError(f"Empty index refers to nonempty JSONL: {index_path}")
    else:
        with jsonl_path.open("rb") as records:
            for offset in {first, previous}:
                if offset:
                    records.seek(offset - 1)
                    if records.read(1) != b"\n":
                        raise ValueError(f"Offset is not a JSONL line boundary: {index_path}")
                records.seek(offset)
                if not isinstance(json.loads(records.readline()), dict):
                    raise ValueError(f"Indexed record is not a JSON object: {jsonl_path}")
    return {"records": record_count, "bytes": byte_count}


def target_paths(source: Path):
    jsonl = source.with_suffix(".jsonl")
    return jsonl, Path(str(jsonl) + ".idx")


def prepare_source(source: Path) -> dict:
    """Build one pair, publishing without overwriting either destination."""
    destinations = target_paths(source)
    for destination in destinations:
        if os.path.lexists(destination):
            raise FileExistsError(f"Refusing to replace existing data: {destination}")
    temporary, published = [], []
    try:
        for destination in destinations:
            fd, name = tempfile.mkstemp(prefix=f".{destination.name}.prepare-", dir=destination.parent)
            os.close(fd)
            temporary.append(Path(name))
        with temporary[0].open("wb") as records, temporary[1].open("w+b") as index:
            index.write(INDEX_HEADER.pack(INDEX_MAGIC, 0, 0))
            count = 0
            for record in iter_json_records(source):
                index.write(OFFSET.pack(records.tell()))
                encoded = (json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
                records.write(encoded)
                count += 1
            byte_count = records.tell()
            index.seek(0)
            index.write(INDEX_HEADER.pack(INDEX_MAGIC, count, byte_count))
        result = check_pair(*temporary)
        # link() is atomic and refuses an existing destination, including a
        # concurrent writer's file. Both temporary files are on the same FS.
        for pending, destination in zip(temporary, destinations):
            os.link(pending, destination)
            published.append(destination)
        return result
    except BaseException:
        # Only files successfully created by this invocation are removed.
        for destination in published:
            destination.unlink()
        raise
    finally:
        for pending in temporary:
            pending.unlink(missing_ok=True)


def configured_sources():
    with DATA_CONFIG.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    entries = config.get("datasets") if isinstance(config, dict) else None
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"No datasets configured in {DATA_CONFIG}")
    seen = set()
    for entry in entries:
        if entry.get("sampling_strategy", "all") != "all":
            raise ValueError("Training preparation requires sampling_strategy: all")
        if not entry.get("json_path") or not entry.get("images_folder"):
            raise ValueError("Each dataset requires json_path and images_folder")
        source = (DATA_CONFIG.parent / entry["json_path"]).resolve()
        images = (DATA_CONFIG.parent / entry["images_folder"]).resolve()
        if source.suffix != ".json":
            raise ValueError(f"Expected a configured JSON array source: {source}")
        if source in seen:
            raise ValueError(f"Duplicate dataset source: {source}")
        if not images.is_dir():
            raise FileNotFoundError(f"Missing images directory: {images}")
        seen.add(source)
        yield source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="Validate existing JSONL/index pairs without writing or reading source JSON arrays")
    args = parser.parse_args()
    sources = list(configured_sources())
    if not args.check_only:
        # Preflight every target before writing any configured source.
        for source in sources:
            if not source.is_file():
                raise FileNotFoundError(source)
            for destination in target_paths(source):
                if os.path.lexists(destination):
                    raise FileExistsError(f"Refusing to replace existing data: {destination}; use --check-only")
    total = 0
    for source in sources:
        result = check_pair(*target_paths(source)) if args.check_only else prepare_source(source)
        total += result["records"]
        print(json.dumps({"source": source.stem, **result}, ensure_ascii=False), flush=True)
    print(json.dumps({"total_records": total, "checked_only": args.check_only}), flush=True)


if __name__ == "__main__":
    main()

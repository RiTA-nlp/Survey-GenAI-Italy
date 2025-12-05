from __future__ import annotations

import dataclasses
import json
import os
import re
from pathlib import Path
from typing import Dict, Iterable

import tyro

PLACEHOLDER_PREFIX = "XXX"
THIS_FILE = Path(__file__).resolve()
IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".bmp",
    ".tiff",
    ".tif",
    ".ico",
    ".svg",
    ".webp",
}


@dataclasses.dataclass
class Args:
    """Command-line arguments for the anonymization script."""

    input_dir: Path
    reverse: bool = False
    config: Path = Path(".anonym")
    map_file: Path = Path(".anonym_map.json")


def load_terms(config_path: Path) -> list[str]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    terms: list[str] = []
    with config_path.open("r", encoding="utf-8") as cfg:
        for line in cfg:
            cleaned = line.strip()
            if cleaned and not cleaned.startswith("#"):
                terms.append(cleaned)
    return terms


def load_mapping(map_path: Path) -> Dict[str, str]:
    if not map_path.exists():
        return {}
    with map_path.open("r", encoding="utf-8") as map_file:
        data = json.load(map_file)
    if not isinstance(data, dict):
        raise ValueError("Mapping file is not a dictionary")
    return {str(k): str(v) for k, v in data.items()}


def save_mapping(map_path: Path, mapping: Dict[str, str]) -> None:
    with map_path.open("w", encoding="utf-8") as map_file:
        json.dump(mapping, map_file, indent=2, sort_keys=True)


def next_placeholder_index(placeholders: Iterable[str]) -> int:
    max_index = 0
    for placeholder in placeholders:
        if placeholder.startswith(PLACEHOLDER_PREFIX):
            suffix = placeholder[len(PLACEHOLDER_PREFIX) :]
            if suffix.isdigit():
                max_index = max(max_index, int(suffix))
    return max_index + 1


def build_replacements(mapping: Dict[str, str], reverse: bool) -> Dict[str, str]:
    if reverse:
        return {v: k for k, v in mapping.items()}
    return dict(mapping)


def build_patterns(replacements: Dict[str, str], ignore_case: bool) -> list[tuple[re.Pattern[str], str]]:
    ordered = sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True)
    flags = re.IGNORECASE if ignore_case else 0
    return [(re.compile(re.escape(source), flags), target) for source, target in ordered]


def iter_files(root: Path, excluded: set[Path]) -> Iterable[Path]:
    skip_dirs = {".git", "__pycache__"}
    for current_dir, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]
        for filename in filenames:
            path = Path(current_dir) / filename
            if path in excluded:
                continue
            if should_skip_file(path):
                continue
            yield path


def should_skip_file(path: Path) -> bool:
    suffix = path.suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        return True
    return is_binary_file(path)


def is_binary_file(path: Path, chunk_size: int = 2048) -> bool:
    try:
        with path.open("rb") as handle:
            chunk = handle.read(chunk_size)
            if not chunk:
                return False
            if b"\0" in chunk:
                return True
            try:
                chunk.decode("utf-8")
            except UnicodeDecodeError:
                return True
    except OSError:
        return True
    return False


def replace_in_file(file_path: Path, patterns: Iterable[tuple[re.Pattern[str], str]]) -> bool:
    try:
        with file_path.open("r", encoding="utf-8", errors="ignore") as handle:
            original = handle.read()
    except Exception:
        return False

    updated = original
    for pattern, target in patterns:
        new_text = pattern.sub(target, updated)
        if new_text != updated:
            updated = new_text

    if updated == original:
        return False

    with file_path.open("w", encoding="utf-8") as handle:
        handle.write(updated)
    return True


def anonymize(args: Args) -> None:
    mapping = load_mapping(args.map_file)
    terms = load_terms(args.config)

    next_index = next_placeholder_index(mapping.values())
    for term in terms:
        if term not in mapping:
            mapping[term] = f"{PLACEHOLDER_PREFIX}{next_index}"
            next_index += 1

    replacements = build_replacements(mapping, reverse=False)
    patterns = build_patterns(replacements, ignore_case=True)
    excluded = {args.config.resolve(), args.map_file.resolve(), THIS_FILE}

    changes = 0
    for file_path in iter_files(args.input_dir, excluded):
        if replace_in_file(file_path, patterns):
            changes += 1

    save_mapping(args.map_file, mapping)
    print(f"Anonymization complete. Updated {changes} files.")


def deanonymize(args: Args) -> None:
    mapping = load_mapping(args.map_file)
    if not mapping:
        raise ValueError("Mapping file missing or empty; cannot perform reverse operation.")

    replacements = build_replacements(mapping, reverse=True)
    patterns = build_patterns(replacements, ignore_case=True)
    excluded = {args.config.resolve(), args.map_file.resolve(), THIS_FILE}

    changes = 0
    for file_path in iter_files(args.input_dir, excluded):
        if replace_in_file(file_path, patterns):
            changes += 1

    print(f"De-anonymization complete. Updated {changes} files.")


def main() -> None:
    args = tyro.cli(Args)
    input_dir = args.input_dir.resolve()
    args.input_dir = input_dir
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

    if args.reverse:
        deanonymize(args)
    else:
        anonymize(args)


if __name__ == "__main__":
    main()

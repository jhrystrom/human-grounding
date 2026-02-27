import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class FileInfo:
    dataset: str
    seed: int
    size: int
    user: str | None = None


def append_english(path: Path) -> Path:
    return path.with_name(path.stem + "_english" + path.suffix)


def find_int_pattern(file_path: Path, pattern: str) -> int | None:
    """
    Find a pattern in the file name.
    """
    match = re.search(pattern, file_path.stem)
    return int(match.group(1)) if match else None


def find_samples(file_path: Path) -> int:
    samples = find_int_pattern(file_path, r"(?:samples_|-n)(\d+)")
    if samples is None:
        raise ValueError(f"Could not find samples in {file_path}")
    return samples


def find_seed(file_path: Path) -> int:
    samples = find_int_pattern(file_path, r"seed[-_]?(\d+)")
    if samples is None:
        raise ValueError(f"Could not find seed in {file_path}")
    return samples


def find_dataset(file_path: Path) -> str:
    dataset_pattern = r"^[a-zA-Z]+"
    found = re.match(dataset_pattern, file_path.stem)
    if found is None:
        raise ValueError(f"Could not find dataset in {file_path}")
    return found.group(0)


def find_user(file_path: Path) -> str | None:
    user_pattern = r"user[-_]?([A-Z]+)"
    found = re.search(user_pattern, file_path.stem)
    if found is None:
        return None
    return found.group(1)


def get_file_info(file_path: Path) -> FileInfo:
    return FileInfo(
        dataset=find_dataset(file_path),
        seed=find_seed(file_path),
        size=find_samples(file_path),
        user=find_user(file_path),
    )

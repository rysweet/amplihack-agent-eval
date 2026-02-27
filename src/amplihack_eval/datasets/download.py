"""Download pre-built learning DB datasets from GitHub Releases.

Philosophy:
- Zero external dependencies (uses urllib + tarfile from stdlib)
- Fail-fast with clear error messages
- Idempotent: re-downloading an existing dataset is a no-op

Public API:
    download_dataset: Download and extract a dataset by name
    list_datasets: List available datasets from GitHub Releases
    load_metadata: Load metadata.json from a local dataset directory
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

GITHUB_REPO = "rysweet/amplihack-agent-eval"
GITHUB_API = f"https://api.github.com/repos/{GITHUB_REPO}/releases"
DATASETS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "datasets"


def _get_datasets_dir() -> Path:
    """Get the datasets directory, creating it if needed."""
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    return DATASETS_DIR


def list_datasets(include_remote: bool = True) -> list[dict[str, Any]]:
    """List available datasets (local and optionally remote).

    Args:
        include_remote: If True, also check GitHub Releases for available datasets.

    Returns:
        List of dataset info dicts with keys: name, local, version, etc.
    """
    datasets: list[dict[str, Any]] = []

    # Local datasets
    ds_dir = _get_datasets_dir()
    for entry in sorted(ds_dir.iterdir()):
        if entry.is_dir() and (entry / "metadata.json").exists():
            meta = load_metadata(entry)
            meta["local"] = True
            meta["path"] = str(entry)
            datasets.append(meta)

    if not include_remote:
        return datasets

    # Remote datasets from GitHub Releases
    local_names = {d["name"] for d in datasets}
    try:
        remote = _fetch_releases()
        for release in remote:
            tag = release.get("tag_name", "")
            if not tag.startswith("dataset-"):
                continue
            name = tag.removeprefix("dataset-")
            if name not in local_names:
                datasets.append({
                    "name": name,
                    "local": False,
                    "tag": tag,
                    "url": release.get("html_url", ""),
                    "published": release.get("published_at", ""),
                })
    except Exception as e:
        logger.warning("Could not fetch remote datasets: %s", e)

    return datasets


def download_dataset(
    name: str,
    output_dir: Path | None = None,
    force: bool = False,
) -> Path:
    """Download and extract a dataset from GitHub Releases.

    Args:
        name: Dataset name (e.g., "5000t-seed42-v1.0")
        output_dir: Where to extract (default: datasets/ in repo root)
        force: Re-download even if already exists locally

    Returns:
        Path to the extracted dataset directory

    Raises:
        RuntimeError: If dataset not found or download fails
    """
    dest_dir = (output_dir or _get_datasets_dir()) / name

    if dest_dir.exists() and not force:
        logger.info("Dataset already exists at %s (use force=True to re-download)", dest_dir)
        return dest_dir

    tag = f"dataset-{name}"

    # Try gh CLI first (handles authentication automatically)
    if _download_with_gh(tag, name, dest_dir):
        return dest_dir

    # Fallback to urllib (works for public repos without auth)
    _download_with_urllib(tag, name, dest_dir)
    return dest_dir


def _download_with_gh(tag: str, name: str, dest_dir: Path) -> bool:
    """Download using gh CLI (preferred, handles auth)."""
    if not shutil.which("gh"):
        return False

    tarball = f"{name}.tar.gz"
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        try:
            result = subprocess.run(
                ["gh", "release", "download", tag,
                 "--repo", GITHUB_REPO,
                 "--pattern", tarball,
                 "--dir", str(tmp_path)],
                capture_output=True, text=True, timeout=120,
            )
            if result.returncode != 0:
                logger.debug("gh download failed: %s", result.stderr)
                return False

            tarball_path = tmp_path / tarball
            if not tarball_path.exists():
                return False

            _extract_tarball(tarball_path, dest_dir)
            logger.info("Downloaded %s via gh CLI -> %s", name, dest_dir)
            return True

        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.debug("gh download error: %s", e)
            return False


def _download_with_urllib(tag: str, name: str, dest_dir: Path) -> None:
    """Download using urllib (fallback for public repos)."""
    tarball = f"{name}.tar.gz"

    # Find the asset URL from the release
    try:
        releases = _fetch_releases()
    except Exception as e:
        raise RuntimeError(f"Cannot fetch releases: {e}") from e

    asset_url = None
    for release in releases:
        if release.get("tag_name") != tag:
            continue
        for asset in release.get("assets", []):
            if asset.get("name") == tarball:
                asset_url = asset.get("browser_download_url")
                break
        break

    if not asset_url:
        raise RuntimeError(
            f"Dataset '{name}' not found in GitHub Releases.\n"
            f"Expected release tag: {tag}\n"
            f"Expected asset: {tarball}\n"
            f"Check available releases: https://github.com/{GITHUB_REPO}/releases"
        )

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp) / tarball
        logger.info("Downloading %s ...", asset_url)

        try:
            req = Request(asset_url, headers={"Accept": "application/octet-stream"})
            with urlopen(req, timeout=300) as resp, open(tmp_path, "wb") as f:
                while True:
                    chunk = resp.read(8192)
                    if not chunk:
                        break
                    f.write(chunk)
        except URLError as e:
            raise RuntimeError(f"Download failed: {e}") from e

        _extract_tarball(tmp_path, dest_dir)
        logger.info("Downloaded %s via urllib -> %s", name, dest_dir)


def _extract_tarball(tarball_path: Path, dest_dir: Path) -> None:
    """Extract a tarball to the destination directory."""
    dest_dir.parent.mkdir(parents=True, exist_ok=True)

    # Clean existing if re-downloading
    if dest_dir.exists():
        shutil.rmtree(dest_dir)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_extract = Path(tmp) / "extract"
        tmp_extract.mkdir()

        with tarfile.open(tarball_path, "r:gz") as tar:
            # Security: check for path traversal
            for member in tar.getmembers():
                if member.name.startswith("/") or ".." in member.name:
                    raise RuntimeError(f"Unsafe path in tarball: {member.name}")
            # Use filter='data' on Python 3.12+ to avoid deprecation warning
            extract_kwargs: dict[str, Any] = {"path": tmp_extract}
            if hasattr(tarfile, "data_filter"):
                extract_kwargs["filter"] = "data"
            tar.extractall(**extract_kwargs)

        # The tarball may contain a top-level directory or files directly
        extracted = list(tmp_extract.iterdir())
        if len(extracted) == 1 and extracted[0].is_dir():
            # Single top-level directory - move it
            shutil.move(str(extracted[0]), str(dest_dir))
        else:
            # Files directly - wrap in dest_dir
            shutil.move(str(tmp_extract), str(dest_dir))


def _fetch_releases() -> list[dict[str, Any]]:
    """Fetch release list from GitHub API."""
    req = Request(GITHUB_API, headers={"Accept": "application/vnd.github.v3+json"})
    try:
        with urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except URLError as e:
        raise RuntimeError(f"Cannot reach GitHub API: {e}") from e


def load_metadata(dataset_path: Path) -> dict[str, Any]:
    """Load metadata.json from a dataset directory.

    Args:
        dataset_path: Path to the dataset directory

    Returns:
        Metadata dict

    Raises:
        FileNotFoundError: If metadata.json does not exist
    """
    meta_path = dataset_path / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"No metadata.json in {dataset_path}")

    with open(meta_path) as f:
        return json.load(f)


__all__ = ["download_dataset", "list_datasets", "load_metadata"]

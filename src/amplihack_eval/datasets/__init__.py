"""Pre-built learning DB datasets for long-horizon evaluation.

Public API:
    download_dataset: Download a dataset from GitHub Releases
    list_datasets: List available datasets from GitHub Releases
    load_metadata: Load metadata.json from a local dataset
"""

from __future__ import annotations

from .download import download_dataset, list_datasets, load_metadata

__all__ = ["download_dataset", "list_datasets", "load_metadata"]

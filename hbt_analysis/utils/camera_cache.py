"""
Camera frame preprocessing cache.

Goal: speed up repeated optimization runs by caching the center-cropped 32x32 frames
as NumPy arrays on disk, so we don't re-read/parse thousands of image files every GA individual.

Supported input formats:
- .tif / .tiff
- .png
"""

from __future__ import annotations

import os
import glob
import tempfile
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
from PIL import Image


def _center_crop_32(img: np.ndarray) -> np.ndarray:
    h, w = img.shape
    if h < 32 or w < 32:
        raise ValueError(f"Image too small to crop to 32x32: got {h}x{w}")
    start_h, start_w = (h - 32) // 2, (w - 32) // 2
    return img[start_h : start_h + 32, start_w : start_w + 32]


def _ensure_parent_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _atomic_save_npy(path: str, arr: np.ndarray) -> None:
    """
    Atomically write a .npy file.

    This prevents concurrent writers (or readers) from observing a partially-written file.
    Strategy: write to a temporary file in the same directory, then os.replace().
    """
    _ensure_parent_dir(path)
    parent = Path(path).parent
    # Ensure suffix is .npy so np.save doesn't auto-append and break atomic replace.
    final_path = Path(path)
    if final_path.suffix != ".npy":
        final_path = final_path.with_suffix(final_path.suffix + ".npy") if final_path.suffix else final_path.with_suffix(".npy")
    tmp_fd, tmp_name = tempfile.mkstemp(prefix=final_path.stem + ".", suffix=".tmp.npy", dir=str(parent))
    os.close(tmp_fd)
    try:
        np.save(tmp_name, arr)
        os.replace(tmp_name, str(final_path))
    finally:
        # If anything went wrong before replace, clean up temp file.
        try:
            if os.path.exists(tmp_name):
                os.remove(tmp_name)
        except OSError:
            pass


def load_center32_frames_full(
    folder_path: str,
    cache_file: Optional[str],
    max_pixel_value: float,
    dtype_on_disk: np.dtype = np.float16,
) -> np.ndarray:
    """
    Load cached full frame stack for a shot, or build it from TIFFs.

    Returns float32 array shaped (n_frames, 32, 32).
    """
    if cache_file and os.path.exists(cache_file):
        arr = np.load(cache_file)
        return arr.astype(np.float32, copy=False)

    frame_files: list[str] = []
    for pat in ("*.tif", "*.tiff", "*.png"):
        frame_files.extend(glob.glob(os.path.join(folder_path, pat)))
    frame_files = sorted(frame_files)
    if not frame_files:
        raise ValueError(f"No frame files found in {folder_path} (expected tif/tiff/png)")

    frames: list[np.ndarray] = []
    for frame_file in frame_files:
        with Image.open(frame_file) as im:
            img = np.array(im, dtype=np.float32) / max_pixel_value
        frames.append(_center_crop_32(img))

    arr = np.asarray(frames, dtype=np.float32)
    if cache_file:
        _atomic_save_npy(cache_file, arr.astype(dtype_on_disk, copy=False))
    return arr


def load_center32_frames_sampled(
    folder_path: str,
    cache_file: Optional[str],
    max_pixel_value: float,
    sample_indices: Sequence[int],
    dtype_on_disk: np.dtype = np.float16,
) -> np.ndarray:
    """
    Load cached sampled frame stack for a shot, or build it by reading only the sampled TIFFs.

    Returns float32 array shaped (len(sample_indices), 32, 32).
    """
    if cache_file and os.path.exists(cache_file):
        arr = np.load(cache_file)
        return arr.astype(np.float32, copy=False)

    frame_files: list[str] = []
    for pat in ("*.tif", "*.tiff", "*.png"):
        frame_files.extend(glob.glob(os.path.join(folder_path, pat)))
    frame_files = sorted(frame_files)
    if not frame_files:
        raise ValueError(f"No frame files found in {folder_path} (expected tif/tiff/png)")

    n = len(frame_files)
    safe_indices = [i for i in sample_indices if 0 <= i < n]
    if len(safe_indices) != len(sample_indices):
        raise ValueError(f"Sample indices out of range for {folder_path}: n={n}, requested={len(sample_indices)}")

    frames: list[np.ndarray] = []
    for i in safe_indices:
        frame_file = frame_files[i]
        with Image.open(frame_file) as im:
            img = np.array(im, dtype=np.float32) / max_pixel_value
        frames.append(_center_crop_32(img))

    arr = np.asarray(frames, dtype=np.float32)
    if cache_file:
        _atomic_save_npy(cache_file, arr.astype(dtype_on_disk, copy=False))
    return arr



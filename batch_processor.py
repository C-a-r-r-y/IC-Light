"""
Batch processing module for IC-Light.

Provides batch image relighting with:
- Multi-file and folder-based input
- tqdm progress bars for console feedback
- Automatic output naming and directory management
- Structured logging of processing parameters and timings
- GPU memory management between images
- Checkpoint/resume support for interrupted batches
- Per-image error isolation (one failure doesn't stop the batch)

Usage:
    from batch_processor import BatchProcessor

    processor = BatchProcessor(output_dir="./output", log_file="./batch.log")
    processor.process_batch(
        image_paths=["img1.jpg", "img2.png"],
        process_fn=my_process_function,
        prompt="sunshine from window",
        ...
    )
"""

from __future__ import annotations

import gc
import json
import logging
import os
import shutil
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logger = logging.getLogger("iclight.batch")
logger.setLevel(logging.INFO)


def _default_logger() -> logging.Logger:
    """Create a console + file logger when none is provided."""
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(ch)
    return logger


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ProcessingRecord:
    """Metadata recorded for a single processed image."""

    input_path: str
    output_path: str
    status: str  # "success", "failed", "skipped"
    width: int = 0
    height: int = 0
    duration_sec: float = 0.0
    prompt: str = ""
    seed: int = 0
    error_message: str = ""
    timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_path": self.input_path,
            "output_path": self.output_path,
            "status": self.status,
            "width": self.width,
            "height": self.height,
            "duration_sec": round(self.duration_sec, 2),
            "prompt": self.prompt,
            "seed": self.seed,
            "error_message": self.error_message,
            "timestamp": self.timestamp,
        }


@dataclass
class BatchManifest:
    """Summary of a completed (or interrupted) batch run."""

    total: int = 0
    succeeded: int = 0
    failed: int = 0
    skipped: int = 0
    total_duration_sec: float = 0.0
    records: List[ProcessingRecord] = field(default_factory=list)
    run_timestamp: str = ""


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def collect_images_from_paths(
    paths: List[str], recursive: bool = False
) -> List[str]:
    """Expand a list of file or directory paths into a flat list of image paths.

    Args:
        paths: List of file or directory paths.
        recursive: If True, walk directories recursively.

    Returns:
        Sorted list of absolute paths to image files.
    """
    result: List[str] = []
    for p in paths:
        p = os.path.abspath(os.path.expanduser(p))
        if os.path.isfile(p):
            ext = os.path.splitext(p)[1].lower()
            if ext in SUPPORTED_FORMATS:
                result.append(p)
            else:
                logger.warning("Skipping unsupported file: %s", p)
        elif os.path.isdir(p):
            for root, dirs, files in os.walk(p):
                for f in sorted(files):
                    ext = os.path.splitext(f)[1].lower()
                    if ext in SUPPORTED_FORMATS:
                        result.append(os.path.join(root, f))
                if not recursive:
                    dirs.clear()  # don't descend
        else:
            logger.warning("Path not found, skipping: %s", p)
    return sorted(set(result))


def make_output_filename(
    input_path: str,
    output_dir: str,
    suffix: str = "_relighted",
    timestamp: bool = True,
    ext: str = ".png",
) -> str:
    """Build a deterministic output path from an input path.

    Naming rule:  {stem}_{suffix}_{timestamp}{ext}

    Args:
        input_path: Source image path.
        output_dir: Target directory (created if needed).
        suffix: Descriptive label appended to the stem.
        timestamp: Include a UTC timestamp in the filename.
        ext: Output extension (default .png).

    Returns:
        Absolute output file path.
    """
    stem = Path(input_path).stem
    ts = ""
    if timestamp:
        ts = "_" + datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{stem}{suffix}{ts}{ext}"
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, filename)


def validate_image(
    image: np.ndarray, min_size: int = 64
) -> Tuple[bool, str]:
    """Check that a numpy image array is valid for processing.

    Args:
        image: NumPy array (H, W, C) or (H, W).
        min_size: Minimum acceptable width/height.

    Returns:
        (is_valid, error_message)
    """
    if image is None:
        return False, "Image is None"
    if not isinstance(image, np.ndarray):
        return False, f"Expected np.ndarray, got {type(image).__name__}"
    if image.ndim < 2 or image.ndim > 3:
        return False, f"Unexpected ndim: {image.ndim}"
    if image.ndim == 3 and image.shape[2] not in (1, 3, 4):
        return False, f"Unexpected channel count: {image.shape[2]}"
    h, w = image.shape[:2]
    if h < min_size or w < min_size:
        return False, f"Image too small ({w}x{h}, min {min_size})"
    if image.size == 0:
        return False, "Image has zero pixels"
    return True, ""


def free_gpu_memory() -> None:
    """Release unreferenced GPU memory after processing an image."""
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _ensure_three_channel(img: np.ndarray) -> np.ndarray:
    """Convert RGBA / greyscale to RGB so downstream code receives 3 channels."""
    if img.ndim == 2:
        return np.stack([img] * 3, axis=-1)
    if img.shape[2] == 4:
        return img[..., :3]
    return img


# ---------------------------------------------------------------------------
# Batch processor
# ---------------------------------------------------------------------------


class BatchProcessor:
    """Orchestrates batch relighting with progress, logging, and recovery.

    Typical usage:

        processor = BatchProcessor(output_dir="./results")
        manifest = processor.process_batch(
            image_paths=["a.jpg", "b.png"],
            process_fn=my_process_relight,
            prompt="golden hour",
            ...
        )
        print(f"Done: {manifest.succeeded}/{manifest.total} succeeded")
    """

    def __init__(
        self,
        output_dir: str = "./batch_output",
        log_file: Optional[str] = None,
        log_level: int = logging.INFO,
    ):
        """Initialize the batch processor.

        Args:
            output_dir: Default directory for processed images.
            log_file: Path to a structured log file (JSONL). If None,
                      './batch_output/batch_log.jsonl' is used.
            log_level: Python logging level for console output.
        """
        self.output_dir = os.path.abspath(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

        if log_file is None:
            log_file = os.path.join(self.output_dir, "batch_log.jsonl")
        self.log_file = os.path.abspath(log_file)

        # Ensure log directory exists
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)

        # Console logger
        self.logger = logging.getLogger(f"iclight.batch.{id(self)}")
        self.logger.setLevel(log_level)
        if not self.logger.handlers:
            ch = logging.StreamHandler()
            ch.setLevel(log_level)
            ch.setFormatter(
                logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
            )
            self.logger.addHandler(ch)

        # Checkpoint file (alongside log)
        self.checkpoint_file = os.path.join(
            os.path.dirname(self.log_file), "batch_checkpoint.json"
        )

        # Runtime state
        self.manifest = BatchManifest()
        self._processed_paths: set = set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_batch(
        self,
        image_paths: List[str],
        process_fn: Callable[..., List[np.ndarray]],
        prompt: str = "",
        image_width: int = 512,
        image_height: int = 640,
        num_samples: int = 1,
        seed: int = 12345,
        steps: int = 25,
        a_prompt: str = "best quality",
        n_prompt: str = "lowres, bad anatomy, bad hands, cropped, worst quality",
        cfg: float = 2.0,
        highres_scale: float = 1.5,
        highres_denoise: float = 0.5,
        lowres_denoise: float = 0.9,
        bg_source: Any = None,
        resume: bool = False,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        **extra_kwargs,
    ) -> BatchManifest:
        """Run relighting on every image in *image_paths*.

        Args:
            image_paths: List of absolute image file paths.
            process_fn: The function that does the actual relighting for one
                        image. It will be called with keyword arguments
                        matching the parameters below.
            prompt: Lighting prompt.
            image_width / image_height: Target resolution.
            num_samples: Number of outputs per input.
            seed: Base random seed.
            steps: Diffusion inference steps.
            a_prompt: Positive prompt suffix.
            n_prompt: Negative prompt.
            cfg: CFG guidance scale.
            highres_scale: High-res pass scale factor.
            highres_denoise: High-res denoising strength.
            lowres_denoise: Low-res denoising strength.
            bg_source: Background source enum value (foreground-conditioned mode)
                       or additional background image (background-conditioned).
            resume: If True, skip images already recorded as successful in the
                    checkpoint file.
            progress_callback: Optional fn(current, total, status_text) called
                               after each image. Suitable for wiring into a
                               Gradio progress component.

        Returns:
            BatchManifest summarizing the run.
        """
        if not image_paths:
            self.logger.warning("No images to process.")
            return self.manifest

        # Load checkpoint for resume
        completed: set = set()
        if resume and os.path.exists(self.checkpoint_file):
            completed = self._load_checkpoint()
            self.logger.info(
                "Resume mode: %d images already completed, skipping.", len(completed)
            )

        # Filter already-completed
        remaining = [p for p in image_paths if p not in completed]
        skipped_count = len(image_paths) - len(remaining)
        if skipped_count > 0:
            self.logger.info("Skipping %d already-processed image(s).", skipped_count)

        total = len(image_paths)
        self.manifest = BatchManifest(
            total=total,
            skipped=skipped_count,
            run_timestamp=datetime.utcnow().isoformat(),
        )

        run_start = time.perf_counter()

        # ---- tqdm progress bar (console) ----
        try:
            from tqdm import tqdm

            pbar = tqdm(
                total=total,
                initial=skipped_count,
                desc="Batch relighting",
                unit="img",
                dynamic_ncols=True,
            )
        except ImportError:
            pbar = None

        for idx, img_path in enumerate(image_paths):
            # Already done via checkpoint
            if img_path in completed:
                continue

            status_text = f"Processing {idx + 1}/{total}: {os.path.basename(img_path)}"
            self.logger.info(status_text)

            if progress_callback is not None:
                progress_callback(idx, total, status_text)

            try:
                record = self._process_one(
                    img_path=img_path,
                    process_fn=process_fn,
                    prompt=prompt,
                    image_width=image_width,
                    image_height=image_height,
                    num_samples=num_samples,
                    seed=seed,
                    steps=steps,
                    a_prompt=a_prompt,
                    n_prompt=n_prompt,
                    cfg=cfg,
                    highres_scale=highres_scale,
                    highres_denoise=highres_denoise,
                    lowres_denoise=lowres_denoise,
                    bg_source=bg_source,
                    **extra_kwargs,
                )
                self.manifest.succeeded += 1
                self._save_checkpoint_entry(img_path)

            except Exception as exc:
                record = ProcessingRecord(
                    input_path=img_path,
                    output_path="",
                    status="failed",
                    prompt=prompt,
                    seed=seed,
                    error_message=f"{type(exc).__name__}: {exc}",
                    timestamp=datetime.utcnow().isoformat(),
                )
                self.manifest.failed += 1
                self.logger.error(
                    "FAILED %s: %s\n%s",
                    os.path.basename(img_path),
                    exc,
                    traceback.format_exc(),
                )

            self.manifest.records.append(record)
            self._write_log_record(record)

            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix_str(
                    f"ok={self.manifest.succeeded} fail={self.manifest.failed}"
                )

            # Free GPU memory between images
            free_gpu_memory()

        if pbar is not None:
            pbar.close()

        self.manifest.total_duration_sec = round(time.perf_counter() - run_start, 2)
        self._write_manifest_summary()

        self.logger.info(
            "Batch complete: %d total, %d succeeded, %d failed, %d skipped in %.1fs",
            self.manifest.total,
            self.manifest.succeeded,
            self.manifest.failed,
            self.manifest.skipped,
            self.manifest.total_duration_sec,
        )

        # Clean checkpoint on full success
        if self.manifest.failed == 0 and os.path.exists(self.checkpoint_file):
            os.remove(self.checkpoint_file)

        return self.manifest

    def create_zip(self, zip_path: Optional[str] = None) -> str:
        """Package all processed images into a zip archive.

        Args:
            zip_path: Target zip file path. Defaults to
                      ``<output_dir>/batch_results_<timestamp>.zip``.

        Returns:
            Path to the created zip file.
        """
        if zip_path is None:
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            zip_path = os.path.join(self.output_dir, f"batch_results_{ts}")

        # shutil.make_archive appends .zip automatically; strip if present
        if zip_path.endswith(".zip"):
            base = zip_path[:-4]
        else:
            base = zip_path

        # Collect output files from records
        files_to_zip = [
            r.output_path
            for r in self.manifest.records
            if r.status == "success" and os.path.isfile(r.output_path)
        ]
        if not files_to_zip:
            self.logger.warning("No successful output files to zip.")
            return ""

        # Create a temporary directory, symlink/copy files, then archive
        tmp_dir = os.path.join(self.output_dir, ".tmp_zip")
        os.makedirs(tmp_dir, exist_ok=True)
        try:
            for f in files_to_zip:
                dest = os.path.join(tmp_dir, os.path.basename(f))
                if not os.path.exists(dest):
                    os.link(f, dest)
            result = shutil.make_archive(base, "zip", tmp_dir)
            self.logger.info("Created zip archive: %s", result)
            return result
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _process_one(
        self,
        img_path: str,
        process_fn: Callable[..., List[np.ndarray]],
        **kwargs,
    ) -> ProcessingRecord:
        """Load one image, run *process_fn*, save results, return record."""
        t0 = time.perf_counter()

        # Load and validate
        img = np.array(Image.open(img_path).convert("RGB"))
        is_ok, err_msg = validate_image(img)
        if not is_ok:
            raise ValueError(f"Invalid image '{img_path}': {err_msg}")

        kwargs["input_fg"] = img
        if "input_bg" in kwargs and kwargs["input_bg"] is not None:
            kwargs["input_bg"] = np.array(
                Image.open(kwargs["input_bg"]).convert("RGB")
            )

        # Run the model
        results = process_fn(**kwargs)

        # Save each output image
        output_paths = []
        if isinstance(results, tuple):
            # Some process functions return (preprocessed_fg, [results])
            # Grab the list part
            flat_results = results[1] if len(results) > 1 else results[0]
        else:
            flat_results = results

        if isinstance(flat_results, np.ndarray):
            flat_results = [flat_results]
        elif not isinstance(flat_results, list):
            flat_results = [flat_results]

        for i, out_img in enumerate(flat_results):
            if not isinstance(out_img, np.ndarray):
                continue
            suffix = f"_relighted_{i}" if len(flat_results) > 1 else "_relighted"
            out_path = make_output_filename(
                img_path,
                self.output_dir,
                suffix=suffix,
            )
            out_pil = Image.fromarray(out_img.astype(np.uint8))
            out_pil.save(out_path)
            output_paths.append(out_path)

        elapsed = time.perf_counter() - t0
        out_path_str = output_paths[0] if output_paths else ""

        return ProcessingRecord(
            input_path=img_path,
            output_path=out_path_str,
            status="success",
            width=img.shape[1],
            height=img.shape[0],
            duration_sec=round(elapsed, 2),
            prompt=kwargs.get("prompt", ""),
            seed=kwargs.get("seed", 0),
            timestamp=datetime.utcnow().isoformat(),
        )

    # ------------------------------------------------------------------
    # Checkpoint / resume
    # ------------------------------------------------------------------

    def _load_checkpoint(self) -> set:
        """Return set of image paths that were already successfully processed."""
        try:
            with open(self.checkpoint_file, "r") as f:
                data = json.load(f)
            return set(data.get("completed", []))
        except (json.JSONDecodeError, KeyError, FileNotFoundError):
            return set()

    def _save_checkpoint_entry(self, img_path: str) -> None:
        """Append *img_path* to the checkpoint file."""
        completed = self._load_checkpoint()
        completed.add(img_path)
        with open(self.checkpoint_file, "w") as f:
            json.dump(
                {"completed": sorted(completed), "updated": datetime.utcnow().isoformat()},
                f,
                indent=2,
            )

    # ------------------------------------------------------------------
    # Logging to file
    # ------------------------------------------------------------------

    def _write_log_record(self, record: ProcessingRecord) -> None:
        """Append one JSON line to the log file."""
        try:
            with open(self.log_file, "a") as f:
                f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
        except OSError as e:
            self.logger.error("Failed to write log record: %s", e)

    def _write_manifest_summary(self) -> None:
        """Write a human-readable summary next to the log file."""
        summary_path = os.path.join(
            os.path.dirname(self.log_file), "batch_summary.json"
        )
        summary = {
            "run_timestamp": self.manifest.run_timestamp,
            "total": self.manifest.total,
            "succeeded": self.manifest.succeeded,
            "failed": self.manifest.failed,
            "skipped": self.manifest.skipped,
            "total_duration_sec": self.manifest.total_duration_sec,
            "output_dir": self.output_dir,
            "records": [r.to_dict() for r in self.manifest.records],
        }
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        self.logger.info("Summary written to %s", summary_path)

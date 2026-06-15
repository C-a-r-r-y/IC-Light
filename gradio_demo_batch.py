"""
IC-Light Batch Processing Demo (Foreground-Conditioned Mode).

Enhanced version of gradio_demo.py with:
- Single-image processing (original functionality preserved)
- Batch multi-file upload and folder import
- Real-time progress bar with ETA
- Automatic output saving with structured naming
- Zip archive download of all results
- Processing log with per-image timing and parameters
- Error isolation (single failure won't stop the batch)
- GPU memory management between images
- Checkpoint/resume for interrupted batches

Usage:
    python gradio_demo_batch.py
"""

from __future__ import annotations

import gc
import os
import tempfile
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import gradio as gr
import numpy as np
import torch

# Re-use the model setup and processing functions from the original demo.
# The `if __name__ == "__main__":` guard in gradio_demo.py prevents
# block.launch() from firing on import.
from gradio_demo import (
    process_relight,
    process,
    run_rmbg,
    resize_and_center_crop,
    resize_without_crop,
    pytorch2numpy,
    numpy2pytorch,
    BGSource,
    quick_prompts,
    quick_subjects,
)

from batch_processor import (
    BatchProcessor,
    collect_images_from_paths,
    free_gpu_memory,
    validate_image,
    SUPPORTED_FORMATS,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPPORTED_EXT_STR = ", ".join(SUPPORTED_FORMATS)


# ---------------------------------------------------------------------------
# Batch wrapper – adapts process_relight for BatchProcessor
# ---------------------------------------------------------------------------

def _batch_process_one(
    input_fg: np.ndarray,
    prompt: str,
    image_width: int,
    image_height: int,
    num_samples: int,
    seed: int,
    steps: int,
    a_prompt: str,
    n_prompt: str,
    cfg: float,
    highres_scale: float,
    highres_denoise: float,
    lowres_denoise: float,
    bg_source: str,
) -> list:
    """Thin wrapper around process_relight that returns a flat list of result images."""
    preprocessed_fg, results = process_relight(
        input_fg=input_fg,
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
    )
    return results


# ---------------------------------------------------------------------------
# Gradio interface
# ---------------------------------------------------------------------------

CSS = """
.batch-status { font-size: 1.1em; padding: 8px; }
.status-pending { color: #888; }
.status-processing { color: #2196F3; font-weight: bold; }
.status-done { color: #4CAF50; font-weight: bold; }
.status-failed { color: #f44336; }
.progress-wrap { margin: 12px 0; }
"""


def _on_batch_process(
    files: list,
    folder_path: str,
    prompt: str,
    image_width: int,
    image_height: int,
    num_samples: int,
    seed: int,
    steps: int,
    a_prompt: str,
    n_prompt: str,
    cfg: float,
    highres_scale: float,
    highres_denoise: float,
    lowres_denoise: float,
    bg_source: str,
    output_dir: str,
    resume: bool,
    progress: gr.Progress = gr.Progress(),
):
    """Handle batch processing button click.

    Collects images from uploaded files and/or folder, then hands them
    off to BatchProcessor for sequential processing with progress tracking.
    """
    # ---- Collect image paths ----
    paths: List[str] = []

    # From multi-file upload
    if files:
        for f in files:
            if hasattr(f, "name"):
                paths.append(os.path.abspath(f.name))
            elif isinstance(f, str):
                paths.append(os.path.abspath(f))

    # From folder
    if folder_path and os.path.isdir(folder_path):
        paths.extend(
            collect_images_from_paths([folder_path], recursive=False)
        )

    # Deduplicate
    paths = sorted(set(paths))

    if not paths:
        yield (
            None,  # gallery
            "",    # zip_path
            "⚠️  No valid images found. Please upload files or specify a folder.",
            "",    # status_html
        )
        return

    # ---- Set up output ----
    if not output_dir.strip():
        output_dir = os.path.join(os.getcwd(), "batch_output")
    os.makedirs(output_dir, exist_ok=True)

    # ---- Initial progress ----
    yield (
        None,
        "",
        f"🔍 Found {len(paths)} image(s). Starting batch...",
        _status_html(0, len(paths), "pending"),
    )

    # ---- Process ----
    processor = BatchProcessor(output_dir=output_dir)

    results_gallery: list = []
    log_lines: list = []

    def progress_callback(current: int, total: int, msg: str):
        """Called by BatchProcessor after each image."""
        nonlocal log_lines
        log_lines.append(f"[{current}/{total}] {msg}")

    try:
        manifest = processor.process_batch(
            image_paths=paths,
            process_fn=_batch_process_one,
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
            resume=resume,
            progress_callback=progress_callback,
        )
    except Exception as exc:
        yield (
            None,
            "",
            f"❌ Batch processing crashed: {exc}\n\n{traceback.format_exc()}",
            _status_html(0, len(paths), "failed"),
        )
        return

    # ---- Collect output paths for gallery ----
    output_paths = [
        r.output_path
        for r in manifest.records
        if r.status == "success" and os.path.isfile(r.output_path)
    ]

    # ---- Create zip ----
    zip_path = ""
    if output_paths:
        try:
            zip_path = processor.create_zip()
        except Exception as e:
            log_lines.append(f"⚠️ Zip creation failed: {e}")

    # ---- Build status HTML ----
    status_html = _status_html(
        manifest.succeeded + manifest.failed,
        manifest.total,
        "done" if manifest.failed == 0 else "done_with_errors",
        manifest=manifest,
    )

    # ---- Build summary text ----
    elapsed = manifest.total_duration_sec
    summary = (
        f"✅ Batch complete!\n"
        f"   Total: {manifest.total} | Succeeded: {manifest.succeeded} | "
        f"Failed: {manifest.failed} | Skipped: {manifest.skipped}\n"
        f"   Duration: {elapsed:.1f}s ({elapsed / max(manifest.total - manifest.skipped, 1):.1f}s per image)\n"
        f"   Output: {output_dir}\n"
    )
    if zip_path:
        summary += f"   Zip: {zip_path}\n"
    summary += f"\n📋 Processing Log:\n" + "\n".join(log_lines[-20:])  # last 20 entries

    yield (
        output_paths if output_paths else None,
        zip_path,
        summary,
        status_html,
    )


def _status_html(current: int, total: int, stage: str, manifest=None) -> str:
    """Build a simple HTML status indicator string."""
    pct = round(current / max(total, 1) * 100)
    stage_label = {
        "pending": "⏳ 待处理 (Pending)",
        "processing": "🔄 处理中 (Processing)",
        "done": "✅ 已完成 (Completed)",
        "done_with_errors": "⚠️ 已完成（有错误）(Completed with errors)",
        "failed": "❌ 失败 (Failed)",
    }.get(stage, stage)

    lines = [
        f'<div class="batch-status status-{stage.split("_")[0]}">',
        f"<strong>{stage_label}</strong><br>",
        f"进度 (Progress): {current} / {total} ({pct}%)",
    ]
    if manifest is not None:
        lines.append(
            f" | 成功 (OK): {manifest.succeeded} | 失败 (Failed): {manifest.failed}"
        )
        if manifest.total_duration_sec:
            lines.append(f"<br>耗时 (Duration): {manifest.total_duration_sec:.1f}s")
    lines.append("</div>")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Build the UI
# ---------------------------------------------------------------------------

def create_demo():
    """Build and return the Gradio Blocks app."""

    with gr.Blocks(css=CSS, title="IC-Light Batch Relighting") as block:
        gr.Markdown(
            """
            ## IC-Light (Relighting with Foreground Condition) — Batch Edition

            This enhanced demo adds **batch processing** on top of the original
            single-image relighting workflow. Upload multiple images or point to a
            folder to process them all with one click.
            """
        )

        with gr.Tabs():
            # ========================================================
            # Tab 1: Single Image (original functionality preserved)
            # ========================================================
            with gr.TabItem("🔆 单张处理 (Single Image)"):
                with gr.Row():
                    with gr.Column():
                        with gr.Row():
                            input_fg = gr.Image(
                                source="upload", type="numpy", label="Image", height=480
                            )
                            output_bg = gr.Image(
                                type="numpy", label="Preprocessed Foreground", height=480
                            )
                        prompt_single = gr.Textbox(label="Prompt")
                        bg_source_single = gr.Radio(
                            choices=[e.value for e in BGSource],
                            value=BGSource.NONE.value,
                            label="Lighting Preference (Initial Latent)",
                            type="value",
                        )
                        with gr.Row():
                            example_quick_subjects = gr.Dataset(
                                samples=quick_subjects,
                                label="Subject Quick List",
                                samples_per_page=1000,
                                components=[prompt_single],
                            )
                            example_quick_prompts = gr.Dataset(
                                samples=quick_prompts,
                                label="Lighting Quick List",
                                samples_per_page=1000,
                                components=[prompt_single],
                            )
                        relight_button = gr.Button(value="Relight", variant="primary")

                        with gr.Group():
                            with gr.Row():
                                num_samples_single = gr.Slider(
                                    label="Images", minimum=1, maximum=12, value=1, step=1
                                )
                                seed_single = gr.Number(label="Seed", value=12345, precision=0)
                            with gr.Row():
                                image_width_single = gr.Slider(
                                    label="Image Width", minimum=256, maximum=1024, value=512, step=64
                                )
                                image_height_single = gr.Slider(
                                    label="Image Height", minimum=256, maximum=1024, value=640, step=64
                                )

                        with gr.Accordion("Advanced options", open=False):
                            steps_single = gr.Slider(label="Steps", minimum=1, maximum=100, value=25, step=1)
                            cfg_single = gr.Slider(label="CFG Scale", minimum=1.0, maximum=32.0, value=2.0, step=0.01)
                            lowres_denoise_single = gr.Slider(
                                label="Lowres Denoise", minimum=0.1, maximum=1.0, value=0.9, step=0.01
                            )
                            highres_scale_single = gr.Slider(
                                label="Highres Scale", minimum=1.0, maximum=3.0, value=1.5, step=0.01
                            )
                            highres_denoise_single = gr.Slider(
                                label="Highres Denoise", minimum=0.1, maximum=1.0, value=0.5, step=0.01
                            )
                            a_prompt_single = gr.Textbox(label="Added Prompt", value="best quality")
                            n_prompt_single = gr.Textbox(
                                label="Negative Prompt",
                                value="lowres, bad anatomy, bad hands, cropped, worst quality",
                            )
                    with gr.Column():
                        result_gallery_single = gr.Gallery(
                            height=832, object_fit="contain", label="Outputs"
                        )

                ips_single = [
                    input_fg,
                    prompt_single,
                    image_width_single,
                    image_height_single,
                    num_samples_single,
                    seed_single,
                    steps_single,
                    a_prompt_single,
                    n_prompt_single,
                    cfg_single,
                    highres_scale_single,
                    highres_denoise_single,
                    lowres_denoise_single,
                    bg_source_single,
                ]
                relight_button.click(
                    fn=process_relight,
                    inputs=ips_single,
                    outputs=[output_bg, result_gallery_single],
                )
                example_quick_prompts.click(
                    lambda x, y: ", ".join(y.split(", ")[:2] + [x[0]]),
                    inputs=[example_quick_prompts, prompt_single],
                    outputs=prompt_single,
                    show_progress=False,
                    queue=False,
                )
                example_quick_subjects.click(
                    lambda x: x[0],
                    inputs=example_quick_subjects,
                    outputs=prompt_single,
                    show_progress=False,
                    queue=False,
                )

            # ====================================================
            # Tab 2: Batch Processing (new features)
            # ====================================================
            with gr.TabItem("📦 批量处理 (Batch Processing)"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 📥 输入 (Input)")

                        batch_files = gr.File(
                            label=f"选择多张图片 (Select Images)",
                            file_count="multiple",
                            file_types=[".jpg", ".jpeg", ".png", ".webp", ".bmp"],
                        )

                        batch_folder = gr.Textbox(
                            label="或输入图片文件夹路径 (Or Folder Path)",
                            placeholder="e.g. /path/to/images/",
                        )

                        prompt_batch = gr.Textbox(
                            label="光照提示词 (Lighting Prompt)",
                            value="sunshine from window",
                        )
                        bg_source_batch = gr.Radio(
                            choices=[e.value for e in BGSource],
                            value=BGSource.NONE.value,
                            label="光照偏好 (Lighting Preference)",
                            type="value",
                        )

                        with gr.Group():
                            with gr.Row():
                                num_samples_batch = gr.Slider(
                                    label="每张生成数 (Samples/Image)",
                                    minimum=1,
                                    maximum=4,
                                    value=1,
                                    step=1,
                                )
                                seed_batch = gr.Number(label="随机种子 (Seed)", value=12345, precision=0)
                            with gr.Row():
                                image_width_batch = gr.Slider(
                                    label="宽度 (Width)", minimum=256, maximum=1024, value=512, step=64
                                )
                                image_height_batch = gr.Slider(
                                    label="高度 (Height)", minimum=256, maximum=1024, value=640, step=64
                                )

                        with gr.Accordion("⚙️ 高级选项 (Advanced)", open=False):
                            steps_batch = gr.Slider(label="Steps", minimum=1, maximum=100, value=25, step=1)
                            cfg_batch = gr.Slider(
                                label="CFG Scale", minimum=1.0, maximum=32.0, value=2.0, step=0.01
                            )
                            lowres_denoise_batch = gr.Slider(
                                label="低分辨率降噪 (Lowres Denoise)",
                                minimum=0.1,
                                maximum=1.0,
                                value=0.9,
                                step=0.01,
                            )
                            highres_scale_batch = gr.Slider(
                                label="高分辨率缩放 (Highres Scale)",
                                minimum=1.0,
                                maximum=3.0,
                                value=1.5,
                                step=0.01,
                            )
                            highres_denoise_batch = gr.Slider(
                                label="高分辨率降噪 (Highres Denoise)",
                                minimum=0.1,
                                maximum=1.0,
                                value=0.5,
                                step=0.01,
                            )
                            a_prompt_batch = gr.Textbox(label="附加提示词 (Added Prompt)", value="best quality")
                            n_prompt_batch = gr.Textbox(
                                label="负向提示词 (Negative Prompt)",
                                value="lowres, bad anatomy, bad hands, cropped, worst quality",
                            )

                        gr.Markdown("### 💾 输出设置 (Output)")

                        output_dir = gr.Textbox(
                            label="输出目录 (Output Directory)",
                            value=os.path.join(os.getcwd(), "batch_output"),
                            placeholder="./batch_output",
                        )

                        with gr.Row():
                            resume_checkbox = gr.Checkbox(
                                label="断点续传 (Resume)",
                                value=False,
                                info="跳过已成功处理的图片",
                            )

                        with gr.Row():
                            batch_button = gr.Button(
                                value="🚀 开始批量处理 (Start Batch)",
                                variant="primary",
                            )

                        # Progress display
                        status_html = gr.HTML(label="处理状态 (Status)")

                    with gr.Column(scale=2):
                        gr.Markdown("### 📊 结果 (Results)")
                        batch_gallery = gr.Gallery(
                            height=500, object_fit="contain", label="处理结果 (Processed Images)"
                        )
                        batch_summary = gr.Textbox(
                            label="处理日志 (Processing Log)",
                            lines=15,
                            max_lines=30,
                            interactive=False,
                        )
                        zip_output = gr.File(label="📥 下载压缩包 (Download Zip)", interactive=False)

                # Wire batch processing
                ips_batch = [
                    batch_files,
                    batch_folder,
                    prompt_batch,
                    image_width_batch,
                    image_height_batch,
                    num_samples_batch,
                    seed_batch,
                    steps_batch,
                    a_prompt_batch,
                    n_prompt_batch,
                    cfg_batch,
                    highres_scale_batch,
                    highres_denoise_batch,
                    lowres_denoise_batch,
                    bg_source_batch,
                    output_dir,
                    resume_checkbox,
                ]
                batch_button.click(
                    fn=_on_batch_process,
                    inputs=ips_batch,
                    outputs=[batch_gallery, zip_output, batch_summary, status_html],
                )

        gr.Markdown(
            """
            ---
            **IC-Light** · ICLR 2025 Oral · [GitHub](https://github.com/lllyasviel/IC-Light)
            """
        )

    return block


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo = create_demo()
    demo.queue(max_size=32)
    demo.launch(server_name="0.0.0.0")

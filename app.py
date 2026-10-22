"""
Gradio Frontend Application (app.py)
Provides a clean, modular web interface for low-VRAM text-to-video generation
using Wan2.1-1.3B-T2V and Lightricks/LTX-Video with optional LLM scene expansion.
"""

import argparse
import logging
import time
from typing import Tuple

import gradio as gr

from director import SceneDirector
from engine import MODEL_REGISTRY, VideoGenerationEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

RESOLUTION_PRESETS = {
    "480x272 (16:9 Low VRAM - Default)": (480, 272),
    "512x320 (16:10 Low VRAM)": (512, 320),
    "640x384 (Balanced)": (640, 384),
    "Custom": None,
}

engine = VideoGenerationEngine()
director = SceneDirector()


def on_preset_change(preset_name: str, current_w: int, current_h: int) -> Tuple[int, int]:
    dims = RESOLUTION_PRESETS.get(preset_name)
    if dims is None:
        return current_w, current_h
    return dims[0], dims[1]


def run_generation_pipeline(
    prompt: str,
    model_key: str,
    num_frames: int,
    width: int,
    height: int,
    steps: int,
    fps: int,
    guidance_scale: float,
    seed: int,
    use_expander: bool,
    api_key: str,
    llm_provider: str,
) -> Tuple[str, str, str]:
    """
    1. Optionally expands the prompt via SceneDirector (with graceful fallback).
    2. Runs the low-VRAM Diffusers pipeline and exports an .mp4 file.
    3. Returns (video_filepath, final_prompt_used, status_message).
    """
    if not prompt or not prompt.strip():
        raise gr.Error("Please enter a scene description prompt.")

    t0 = time.time()
    final_prompt = director.enhance_prompt(
        prompt=prompt,
        use_enhancer=use_expander,
        api_key=api_key,
        provider=llm_provider,
    )

    video_path, meta = engine.generate_video(
        prompt=final_prompt,
        model_key=model_key,
        num_frames=int(num_frames),
        width=int(width),
        height=int(height),
        num_inference_steps=int(steps),
        fps=int(fps),
        guidance_scale=float(guidance_scale),
        seed=int(seed),
    )

    elapsed = time.time() - t0
    status = (
        f"Generated {meta['num_frames']} frames at {meta['width']}x{meta['height']} "
        f"({meta['fps']} FPS, {meta['steps']} steps) on {engine.device.upper()} "
        f"in {elapsed:.1f}s."
    )
    return video_path, final_prompt, status


def build_ui() -> gr.Blocks:
    model_choices = list(MODEL_REGISTRY.keys())

    with gr.Blocks(title="Zero-Cost AI Video Studio") as demo:
        gr.Markdown(
            """
            # Zero-Cost AI Video Studio
            Lightweight, low-VRAM Text-to-Video generation powered by **Wan2.1-1.3B-T2V** and **Lightricks/LTX-Video** (`< 12–14 GB VRAM` with CPU offload & VAE slicing).
            """
        )

        with gr.Row():
            with gr.Column(scale=5):
                prompt_input = gr.Textbox(
                    label="Scene Description Prompt",
                    placeholder="e.g., A lone astronaut walking across a bioluminescent alien beach at twilight, gentle waves glowing neon blue...",
                    lines=4,
                )

                model_dropdown = gr.Dropdown(
                    choices=model_choices,
                    value="Wan2.1-1.3B-T2V",
                    label="Diffusion Model",
                )

                with gr.Group():
                    gr.Markdown("### Generation Parameters")
                    res_preset = gr.Dropdown(
                        choices=list(RESOLUTION_PRESETS.keys()),
                        value="480x272 (16:9 Low VRAM - Default)",
                        label="Resolution Preset",
                    )
                    with gr.Row():
                        width_slider = gr.Slider(
                            minimum=256,
                            maximum=832,
                            step=16,
                            value=480,
                            label="Width (px)",
                        )
                        height_slider = gr.Slider(
                            minimum=256,
                            maximum=480,
                            step=16,
                            value=272,
                            label="Height (px)",
                        )

                    with gr.Row():
                        frames_slider = gr.Slider(
                            minimum=9,
                            maximum=49,
                            step=1,
                            value=25,
                            label="Frame Count (16 or 25 recommended)",
                        )
                        fps_slider = gr.Slider(
                            minimum=4,
                            maximum=24,
                            step=1,
                            value=8,
                            label="FPS",
                        )

                    with gr.Row():
                        steps_slider = gr.Slider(
                            minimum=10,
                            maximum=50,
                            step=1,
                            value=20,
                            label="Inference Steps",
                        )
                        guidance_slider = gr.Slider(
                            minimum=1.0,
                            maximum=12.0,
                            step=0.5,
                            value=5.0,
                            label="Guidance Scale (CFG)",
                        )

                    seed_input = gr.Number(
                        value=-1,
                        precision=0,
                        label="Seed (-1 for random)",
                    )

                with gr.Accordion("LLM Scene Expander (Optional)", open=False):
                    use_expander_toggle = gr.Checkbox(
                        value=False,
                        label="Use LLM Scene Expander",
                        info="Automatically enriches your prompt with cinematic lighting, camera movement, and motion cues. Falls back to raw prompt if no key is provided.",
                    )
                    with gr.Row():
                        llm_provider_dropdown = gr.Dropdown(
                            choices=["auto", "openai", "gemini", "anthropic"],
                            value="auto",
                            label="Provider",
                            scale=1,
                        )
                        api_key_input = gr.Textbox(
                            label="API Key (OpenAI / Gemini / Anthropic)",
                            placeholder="sk-... or AIza... (leave blank to use env var or passthrough)",
                            type="password",
                            scale=3,
                        )

                generate_btn = gr.Button("Generate Video", variant="primary", size="lg")

            with gr.Column(scale=5):
                video_output = gr.Video(
                    label="Generated Video (.mp4)",
                    format="mp4",
                    interactive=False,
                )
                expanded_prompt_output = gr.Textbox(
                    label="Active Diffusion Prompt (After Director Pass)",
                    lines=4,
                    interactive=False,
                )
                status_output = gr.Textbox(
                    label="Engine Status",
                    value=f"Ready | Detected Backend: {engine.device.upper()} ({engine.dtype})",
                    interactive=False,
                )

        res_preset.change(
            fn=on_preset_change,
            inputs=[res_preset, width_slider, height_slider],
            outputs=[width_slider, height_slider],
        )

        generate_btn.click(
            fn=run_generation_pipeline,
            inputs=[
                prompt_input,
                model_dropdown,
                frames_slider,
                width_slider,
                height_slider,
                steps_slider,
                fps_slider,
                guidance_slider,
                seed_input,
                use_expander_toggle,
                api_key_input,
                llm_provider_dropdown,
            ],
            outputs=[video_output, expanded_prompt_output, status_output],
        )

    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Zero-Cost AI Video Studio (Gradio + Diffusers)")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share tunnel")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server bind address")
    parser.add_argument("--port", type=int, default=7860, help="Server port")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    app = build_ui()
    app.queue().launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )

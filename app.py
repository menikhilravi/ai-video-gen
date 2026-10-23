"""
Gradio Frontend Application (app.py)
Provides a production-grade "Grok Imagine"-style multi-stage interface:
  [1/4] Generating Keyframe (FLUX.1-schnell / SDXL-Turbo or Direct Image Upload)
  [2/4] Animating Frames (Lightricks/LTX-Video I2V/T2V or Wan2.1-1.3B)
  [3/4] Synthesizing Sound (AudioLDM 2 Ambient Foley & Sound FX)
  [4/4] Muxing Output (FFmpeg synchronized H.264 + AAC MP4)
"""

import argparse
import logging
import os
import time
from typing import Optional, Tuple

# Disable Gradio external telemetry & Hugging Face Xet RAM-spiking chunk reconstruction
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

import gradio as gr
from PIL import Image

from director import SceneDirector
from engine import (
    AUDIO_MODEL_REGISTRY,
    KEYFRAME_MODEL_REGISTRY,
    MODEL_REGISTRY,
    VideoGenerationEngine,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

INPUT_MODES = [
    "Text-to-Video with Auto-Keyframe (FLUX / SDXL -> I2V)",
    "Direct Image Upload to Video (Upload Photo -> I2V)",
    "Direct Text-to-Video (Skip Keyframe Stage)",
]

RESOLUTION_PRESETS = {
    "832x480 (16:9 Native 480p HD - Best Quality)": (832, 480),
    "768x512 (3:2 Cinematic I2V)": (768, 512),
    "640x384 (16:10 Balanced)": (640, 384),
    "512x320 (16:10 Fast)": (512, 320),
    "480x272 (16:9 Ultra-Fast Draft)": (480, 272),
    "Custom": None,
}

engine = VideoGenerationEngine()
director = SceneDirector()


def on_preset_change(preset_name: str, current_w: int, current_h: int) -> Tuple[int, int]:
    dims = RESOLUTION_PRESETS.get(preset_name)
    if dims is None:
        return current_w, current_h
    return dims[0], dims[1]


def on_input_mode_change(mode: str):
    is_upload = "Direct Image Upload" in mode
    is_auto_kf = "Auto-Keyframe" in mode
    return (
        gr.update(visible=is_upload),
        gr.update(visible=is_auto_kf),
    )


def on_audio_toggle_change(enabled: bool):
    return (
        gr.update(visible=enabled),
        gr.update(visible=enabled),
    )


def run_generation_pipeline(
    prompt: str,
    input_mode: str,
    uploaded_image: Optional[Image.Image],
    keyframe_model_key: str,
    model_key: str,
    num_frames: int,
    width: int,
    height: int,
    steps: int,
    fps: int,
    guidance_scale: float,
    seed: int,
    generate_audio: bool,
    custom_audio_prompt: str,
    audio_model_key: str,
    use_expander: bool,
    api_key: str,
    llm_provider: str,
    progress=gr.Progress(track_tqdm=True),
):
    """
    Generator function that yields live stage progression updates to the Gradio UI:
    [1/4] Generating Keyframe -> [2/4] Animating Frames -> [3/4] Synthesizing Sound -> [4/4] Muxing Output
    """
    if not prompt or not prompt.strip():
        raise gr.Error("Please enter a scene description prompt.")

    if "Direct Image Upload" in input_mode and uploaded_image is None:
        raise gr.Error("Please upload an anchor photo or switch to 'Text-to-Video with Auto-Keyframe'.")

    t0 = time.time()
    pipeline_banner = (
        "[1/4] Generating Keyframe -> [2/4] Animating Frames -> "
        "[3/4] Synthesizing Sound -> [4/4] Muxing Output"
    )

    progress(0.02, desc="Expanding visual & Foley audio prompts...")
    final_visual_prompt = director.enhance_prompt(
        prompt=prompt,
        use_enhancer=use_expander,
        api_key=api_key,
        provider=llm_provider,
    )
    final_audio_prompt = director.derive_audio_prompt(
        visual_prompt=final_visual_prompt,
        custom_audio_prompt=custom_audio_prompt,
    )

    initial_log = (
        f"Pipeline Progression: {pipeline_banner}\n"
        f"────────────────────────────────────────────────────────────\n"
        f"⏳ Starting 4-Stage Sequential Pipeline on {engine.device.upper()} ({engine.dtype})..."
    )
    yield None, uploaded_image, final_visual_prompt, final_audio_prompt, initial_log

    latest_log_box = [initial_log]

    def _update_status(stage_text: str):
        latest_log_box[0] = (
            f"Pipeline Progression: {pipeline_banner}\n"
            f"────────────────────────────────────────────────────────────\n"
            f"{stage_text}"
        )

    final_video_path, keyframe_img, stage_summary = engine.run_multistage_pipeline(
        prompt=final_visual_prompt,
        input_mode=input_mode,
        uploaded_image=uploaded_image,
        keyframe_model_key=keyframe_model_key,
        model_key=model_key,
        num_frames=int(num_frames),
        width=int(width),
        height=int(height),
        num_inference_steps=int(steps),
        fps=int(fps),
        guidance_scale=float(guidance_scale),
        seed=int(seed),
        generate_audio=bool(generate_audio),
        audio_prompt=final_audio_prompt,
        audio_model_key=audio_model_key,
        progress_callback=progress,
        status_callback=_update_status,
    )

    elapsed = time.time() - t0
    completed_log = (
        f"Pipeline Progression: {pipeline_banner}\n"
        f"────────────────────────────────────────────────────────────\n"
        f"{stage_summary}\n"
        f"────────────────────────────────────────────────────────────\n"
        f"🎬 Completed in {elapsed:.1f}s | Peak VRAM kept < 15 GB via sequential offloading."
    )

    yield final_video_path, keyframe_img, final_visual_prompt, final_audio_prompt, completed_log


def build_ui() -> gr.Blocks:
    model_choices = list(MODEL_REGISTRY.keys())
    kf_choices = list(KEYFRAME_MODEL_REGISTRY.keys())
    audio_choices = list(AUDIO_MODEL_REGISTRY.keys())

    with gr.Blocks(title="Grok Imagine Studio — Multi-Stage AI Video + Foley Audio") as demo:
        gr.Markdown(
            """
            # 🎬 Grok Imagine Studio — 4-Stage Photorealistic Video & Foley Audio Pipeline
            **`[1/4] Generating Keyframe (FLUX.1-schnell / SDXL-Turbo)` $\\rightarrow$ `[2/4] Animating Frames (LTX-Video / Wan2.1 I2V)` $\\rightarrow$ `[3/4] Synthesizing Sound (AudioLDM 2)` $\\rightarrow$ `[4/4] Muxing Output (FFmpeg AAC+H.264)`**
            """
        )

        with gr.Row():
            with gr.Column(scale=5):
                input_mode_radio = gr.Radio(
                    choices=INPUT_MODES,
                    value=INPUT_MODES[0],
                    label="Stage 1 Input Mode",
                )

                uploaded_image_input = gr.Image(
                    label="Upload Anchor Image (for Direct Image-to-Video)",
                    type="pil",
                    visible=False,
                )

                keyframe_model_dropdown = gr.Dropdown(
                    choices=kf_choices,
                    value=kf_choices[0],
                    label="Stage 1 Keyframe Synthesizer (Text-to-Image)",
                    visible=True,
                )

                prompt_input = gr.Textbox(
                    label="Scene Description Prompt (Use '||' between shots for multi-shot stories)",
                    placeholder="e.g., A lone astronaut walking across a bioluminescent alien beach at twilight, gentle waves glowing neon blue...",
                    lines=3,
                )

                model_dropdown = gr.Dropdown(
                    choices=model_choices,
                    value="Lightricks/LTX-Video",
                    label="Stage 2 Video Animation Backbone (LTX-Video supports native I2V + T2V on 15GB T4)",
                )

                with gr.Group():
                    gr.Markdown("### Stage 3: Ambient Sound FX & Foley Synthesis")
                    generate_audio_checkbox = gr.Checkbox(
                        value=True,
                        label="Generate Matching Audio (Synchronized Foley & Ambience)",
                    )
                    audio_prompt_input = gr.Textbox(
                        label="Custom Sound / Foley Cues (Optional — auto-populated from visual prompt if left blank)",
                        placeholder="Leave blank to auto-derive Foley cues (e.g., ocean waves, keyboard clicks, city ambience, footsteps)...",
                        lines=2,
                        visible=True,
                    )
                    audio_model_dropdown = gr.Dropdown(
                        choices=audio_choices,
                        value=audio_choices[0],
                        label="Audio Foley Model",
                        visible=True,
                    )

                with gr.Accordion("Stage 2 Video Generation Parameters", open=False):
                    res_preset = gr.Dropdown(
                        choices=list(RESOLUTION_PRESETS.keys()),
                        value="768x512 (3:2 Cinematic I2V)",
                        label="Resolution Preset",
                    )
                    with gr.Row():
                        width_slider = gr.Slider(
                            minimum=256,
                            maximum=832,
                            step=16,
                            value=768,
                            label="Width (px)",
                        )
                        height_slider = gr.Slider(
                            minimum=256,
                            maximum=512,
                            step=16,
                            value=512,
                            label="Height (px)",
                        )

                    with gr.Row():
                        frames_slider = gr.Slider(
                            minimum=17,
                            maximum=81,
                            step=4,
                            value=49,
                            label="Frames Per Shot (49 = ~3s, 81 = ~5s per shot)",
                        )
                        fps_slider = gr.Slider(
                            minimum=8,
                            maximum=24,
                            step=1,
                            value=16,
                            label="FPS",
                        )

                    with gr.Row():
                        steps_slider = gr.Slider(
                            minimum=15,
                            maximum=50,
                            step=1,
                            value=30,
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
                        value=True,
                        label="Use LLM Scene Expander",
                        info="Automatically enriches your prompt with camera direction, lighting, and Foley cues (uses zero-key local cinematic rewriter if no API key is entered).",
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
                            placeholder="sk-... or AIza... (leave blank for built-in local rewriter)",
                            type="password",
                            scale=3,
                        )

                generate_btn = gr.Button(
                    "🚀 Run 4-Stage Pipeline (Keyframe -> Video -> Foley -> Mux)",
                    variant="primary",
                    size="lg",
                )

            with gr.Column(scale=5):
                video_output = gr.Video(
                    label="Stage 4 Muxed Video + Audio Output (.mp4)",
                    format="mp4",
                    interactive=False,
                )
                keyframe_output = gr.Image(
                    label="Stage 1 Anchor Keyframe Preview",
                    type="pil",
                    interactive=False,
                )
                status_output = gr.Textbox(
                    label="Multi-Stage Pipeline Progression Log",
                    value=(
                        "[1/4] Generating Keyframe -> [2/4] Animating Frames -> "
                        "[3/4] Synthesizing Sound -> [4/4] Muxing Output\n"
                        f"Ready | Backend: {engine.device.upper()} ({engine.dtype}) | "
                        f"Host RAM: {engine.system_ram_gb:.1f} GB"
                    ),
                    lines=6,
                    interactive=False,
                )
                with gr.Accordion("Active Visual & Audio Prompts (After Director Pass)", open=False):
                    expanded_prompt_output = gr.Textbox(
                        label="Active Visual Diffusion Prompt",
                        lines=3,
                        interactive=False,
                    )
                    derived_audio_prompt_output = gr.Textbox(
                        label="Active Audio Foley Prompt",
                        lines=2,
                        interactive=False,
                    )

        input_mode_radio.change(
            fn=on_input_mode_change,
            inputs=[input_mode_radio],
            outputs=[uploaded_image_input, keyframe_model_dropdown],
        )

        generate_audio_checkbox.change(
            fn=on_audio_toggle_change,
            inputs=[generate_audio_checkbox],
            outputs=[audio_prompt_input, audio_model_dropdown],
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
                input_mode_radio,
                uploaded_image_input,
                keyframe_model_dropdown,
                model_dropdown,
                frames_slider,
                width_slider,
                height_slider,
                steps_slider,
                fps_slider,
                guidance_slider,
                seed_input,
                generate_audio_checkbox,
                audio_prompt_input,
                audio_model_dropdown,
                use_expander_toggle,
                api_key_input,
                llm_provider_dropdown,
            ],
            outputs=[
                video_output,
                keyframe_output,
                expanded_prompt_output,
                derived_audio_prompt_output,
                status_output,
            ],
        )

    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grok Imagine Studio — 4-Stage AI Video + Audio")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share tunnel")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server bind address")
    parser.add_argument("--port", type=int, default=7860, help="Server port")
    parser.add_argument(
        "--auth",
        type=str,
        default=os.environ.get("GRADIO_AUTH", ""),
        help="Optional username:password to lock the Gradio UI (e.g., --auth admin:secret123)",
    )
    return parser.parse_args()


def _parse_auth(auth_str: str) -> Optional[Tuple[str, str]]:
    cleaned = (auth_str or "").strip()
    if not cleaned:
        return None
    if ":" not in cleaned:
        raise ValueError("--auth must be in 'username:password' format (e.g., --auth nikhil:mysecret)")
    user, pwd = cleaned.split(":", 1)
    return user.strip(), pwd.strip()


if __name__ == "__main__":
    args = parse_args()
    auth_tuple = _parse_auth(args.auth)
    app = build_ui()
    app.queue().launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        auth=auth_tuple,
    )

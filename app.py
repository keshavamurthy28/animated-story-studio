import asyncio
import json
import os
import tempfile
import time
from io import BytesIO

import edge_tts
import fal_client
import requests
import streamlit as st
from google import genai
from google.genai import types
from huggingface_hub import InferenceClient
from PIL import Image

try:
    from moviepy.editor import AudioFileClip, ImageClip, VideoFileClip, concatenate_videoclips
    MOVIEPY_V2 = False
except ModuleNotFoundError:
    from moviepy import AudioFileClip, ImageClip, VideoFileClip, concatenate_videoclips
    MOVIEPY_V2 = True

st.set_page_config(
    page_title="AI Animated Story Studio",
    page_icon="🎬",
    layout="centered",
)

for key, default in [
    ("scenes", []),
    ("images", []),
    ("scene_clips", []),
    ("audio_files", []),
]:
    if key not in st.session_state:
        st.session_state[key] = default


def run_async(coro):
    """Safely run async coroutines inside Streamlit's event loop."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            with tempfile.NamedTemporaryFile(suffix=".py") as _:
                pass
            new_loop = asyncio.new_event_loop()
            try:
                return new_loop.run_until_complete(coro)
            finally:
                new_loop.close()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


def extract_video_url(result):
    """Handle common fal.ai response shapes safely."""
    if isinstance(result, dict):
        if "video" in result and isinstance(result["video"], dict) and "url" in result["video"]:
            return result["video"]["url"]
        if "data" in result and isinstance(result["data"], dict):
            data = result["data"]
            if "video" in data and isinstance(data["video"], dict) and "url" in data["video"]:
                return data["video"]["url"]
    raise ValueError(f"Unexpected video response format: {result}")


def apply_duration(clip, duration):
    return clip.with_duration(duration) if MOVIEPY_V2 else clip.set_duration(duration)


def apply_audio(clip, audio):
    return clip.with_audio(audio) if MOVIEPY_V2 else clip.set_audio(audio)


def apply_subclip(clip, start_t, end_t):
    return clip.subclipped(start_t, end_t) if MOVIEPY_V2 else clip.subclip(start_t, end_t)


def fit_image_to_frame(img_path, target_w, target_h):
    """Resize and crop an image cleanly to exact (target_w, target_h) using Pillow."""
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    with Image.open(img_path) as im:
        rgb = im.convert("RGB")
        scale = max(target_w / rgb.width, target_h / rgb.height)
        new_w = max(1, int(round(rgb.width * scale)))
        new_h = max(1, int(round(rgb.height * scale)))
        resized = rgb.resize((new_w, new_h), resampling)
        left = max(0, (new_w - target_w) // 2)
        top = max(0, (new_h - target_h) // 2)
        cropped = resized.crop((left, top, left + target_w, top + target_h))
        fitted_path = f"fitted_{os.path.basename(img_path)}"
        cropped.save(fitted_path, format="JPEG", quality=92)
        return fitted_path


def generate_scene_image_gemini(gemini_key, prompt_text, ratio_str):
    """Generate an image using Gemini image models via google-genai."""
    client = genai.Client(api_key=gemini_key.strip())
    image_models = ["gemini-3.1-flash-lite-image", "gemini-3.1-flash-image"]
    last_err = None

    for model_id in image_models:
        try:
            response = client.models.generate_content(
                model=model_id,
                contents=prompt_text,
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE"],
                    image_config=types.ImageConfig(aspect_ratio=ratio_str),
                ),
            )
            candidates = getattr(response, "candidates", None) or []
            for cand in candidates:
                content = getattr(cand, "content", None)
                parts = getattr(content, "parts", None) or []
                for part in parts:
                    inline = getattr(part, "inline_data", None)
                    if inline and getattr(inline, "data", None):
                        return Image.open(BytesIO(inline.data)).convert("RGB")
            raise ValueError(f"No inline image returned by {model_id}")
        except Exception as exc:
            last_err = exc
    raise RuntimeError(f"Gemini image generation failed: {last_err}")


def generate_scene_image_hf(hf_token, prompt_text, target_w, target_h):
    """Generate an image via Hugging Face InferenceClient."""
    client = InferenceClient(token=hf_token.strip())
    hf_models = [
        "black-forest-labs/FLUX.1-schnell",
        "stabilityai/sdxl-turbo",
    ]
    last_err = None

    for model_id in hf_models:
        try:
            pil_img = client.text_to_image(
                prompt=prompt_text,
                model=model_id,
                width=target_w,
                height=target_h,
            )
            return pil_img.convert("RGB")
        except Exception as exc:
            last_err = exc
    raise RuntimeError(f"Hugging Face image generation failed: {last_err}")


st.title("🎬 AI Animated Story Studio")
st.caption("Turn rough story ideas into downloadable animated videos.")

with st.sidebar:
    st.header("🔑 Studio Settings")
    api_key = st.text_input("Gemini API Key", type="password")
    hf_token = st.text_input("Hugging Face Token (free image option)", type="password")
    image_provider = st.selectbox(
        "Free Scene Image Engine",
        [
            "Gemini Image (uses Gemini API Key)",
            "Hugging Face Inference (uses HF Token)",
        ],
    )
    fal_key = st.text_input("fal.ai API Key (optional paid I2V)", type="password")
    i2v_model = st.selectbox(
        "Image-to-Video Model (Paid fal.ai)",
        [
            "fal-ai/wan-3.0-prime/image-to-video",
            "fal-ai/minimax/h3/image-to-video",
            "fal-ai/kling-video/v3/standard/image-to-video",
            "bytedance/seedance-2.0/image-to-video",
        ],
    )
    aspect_ratio = st.selectbox("Video Format", ["9:16 (Reels / Shorts)", "16:9 (YouTube)"])
    if "9:16" in aspect_ratio:
        width, height, ratio_str = 720, 1280, "9:16"
    else:
        width, height, ratio_str = 1280, 720, "16:9"

tab1, tab2, tab3, tab4 = st.tabs([
    "1️⃣ Rough Story",
    "2️⃣ Characters & Clips",
    "3️⃣ Voice & Sound",
    "4️⃣ Download Video",
])

# --- TAB 1: SCRIPT WRITER ---
with tab1:
    st.subheader("Write Your Rough Idea")
    rough_story = st.text_area(
        "Type your rough story in any style (even broken sentences):",
        placeholder="e.g., 4 hostel boys walk near a forest village, a giant bear appears, two climb a tree, one lies still holding breath, and learns who a true friend is.",
        height=140,
    )
    num_scenes = st.slider("Number of Scenes", 3, 6, 4)

    if st.button("✨ Polish Story & Create Scenes", type="primary"):
        if not api_key or not rough_story.strip():
            st.error("Please enter your Gemini API Key in the sidebar and write a rough story.")
        else:
            fallback_models = [
                "gemini-3.8-flash",
                "gemini-3.6-flash",
                "gemini-3.5-flash-lite",
                "gemini-3.1-flash-lite",
            ]
            prompt = f"""
            Fix and rewrite this rough story into an engaging {num_scenes}-scene animated short script.
            Return ONLY valid JSON as a list of objects, with no markdown formatting.
            Each object must have:
            - "narration": clear, emotional voiceover text (1-2 sentences)
            - "visual": detailed visual action description for that scene (no character appearance details, just action and background)
            Rough story: {rough_story.strip()}
            """
            client = genai.Client(api_key=api_key.strip())
            last_exc = None
            parsed_scenes = None

            with st.spinner("Polishing grammar and structuring scenes..."):
                for model_name in fallback_models:
                    for attempt in range(2):
                        try:
                            response = client.models.generate_content(
                                model=model_name,
                                contents=prompt,
                            )
                            raw_text = (response.text or "").strip()
                            if raw_text.startswith("```"):
                                lines = [
                                    line for line in raw_text.splitlines()
                                    if not line.strip().startswith("```")
                                ]
                                raw_text = "\n".join(lines).strip()

                            candidate = json.loads(raw_text)
                            if isinstance(candidate, list) and len(candidate) > 0:
                                parsed_scenes = candidate
                                break
                        except Exception as exc:
                            last_exc = exc
                            msg = str(exc)
                            if "503" in msg or "UNAVAILABLE" in msg or "429" in msg:
                                time.sleep(1.5 * (attempt + 1))
                                continue
                            break
                    if parsed_scenes is not None:
                        break

            if parsed_scenes is not None:
                st.session_state.scenes = [
                    {
                        "narration": str(item.get("narration", "")).strip(),
                        "visual": str(item.get("visual", "")).strip(),
                    }
                    for item in parsed_scenes
                ]
                st.success("Script ready! Review below or move to Tab 2.")
            else:
                st.error(f"Could not generate script after retries/fallbacks: {last_exc}")

    if st.session_state.scenes:
        for i, scene in enumerate(st.session_state.scenes):
            with st.expander(f"Scene {i + 1}", expanded=True):
                scene["narration"] = st.text_area(
                    f"Narration {i + 1}",
                    value=scene["narration"],
                    key=f"nar_{i}",
                )
                scene["visual"] = st.text_input(
                    f"Visual Action {i + 1}",
                    value=scene["visual"],
                    key=f"vis_{i}",
                )

# --- TAB 2: CHARACTER SELECTION, ART & MOVING CLIPS ---
with tab2:
    st.subheader("Cast Your Characters & Art Style")
    art_style = st.selectbox(
        "Choose Animation Style",
        [
            "3D Pixar Disney animation style, vibrant lighting, expressive faces",
            "2D Studio Ghibli anime style, soft watercolor backgrounds, warm",
            "Children's storybook illustration, textured paper, magical glow",
            "3D Unreal Engine cinematic fantasy animation",
        ],
    )
    char_desc = st.text_input(
        "Describe your main character(s) (keeps them looking the same in every scene):",
        placeholder="e.g., Four teenage boys aged 12-15 in hostel uniforms and a large brown forest bear",
    )

    if st.button("🎨 Generate Scene Artwork", type="primary"):
        if not st.session_state.scenes:
            st.warning("Please generate your script in Tab 1 first.")
        elif image_provider.startswith("Gemini") and not api_key.strip():
            st.error("Enter your Gemini API Key in the sidebar (or switch to Hugging Face Inference).")
        elif image_provider.startswith("Hugging Face") and not hf_token.strip():
            st.error("Enter your Hugging Face Token in the sidebar (or switch to Gemini Image).")
        else:
            st.session_state.images = []
            st.session_state.scene_clips = []
            progress = st.progress(0)

            try:
                for idx, scene in enumerate(st.session_state.scenes):
                    full_prompt = (
                        f"{art_style}. Consistent characters: {char_desc}. "
                        f"Scene action: {scene['visual']}"
                    ).strip()

                    if image_provider.startswith("Gemini"):
                        pil_img = generate_scene_image_gemini(api_key, full_prompt, ratio_str)
                    else:
                        pil_img = generate_scene_image_hf(hf_token, full_prompt, width, height)

                    img_path = f"scene_{idx}.png"
                    pil_img.save(img_path, format="PNG")
                    st.session_state.images.append(img_path)
                    progress.progress((idx + 1) / len(st.session_state.scenes))

                st.success(f"Generated all {len(st.session_state.images)} scene images!")
            except Exception as exc:
                st.error(f"Error generating scene images: {exc}")

    if st.session_state.images:
        cols = st.columns(2)
        for idx, img_path in enumerate(st.session_state.images):
            if os.path.exists(img_path):
                cols[idx % 2].image(img_path, caption=f"Scene {idx + 1}", use_container_width=True)

        st.divider()
        st.subheader("Optional: Paid Image-to-Video Clips (fal.ai)")
        motion_hint = st.text_input(
            "Character & Camera Motion Hint",
            value="Expressive character movement, natural blinking and gesture, smooth cinematic camera motion",
        )

        if st.button("🎞️ Animate Scene Images into Clips"):
            if not fal_key.strip():
                st.warning("fal.ai requires credits and an API key. Leave this empty to render free videos in Tab 4.")
            else:
                os.environ["FAL_KEY"] = fal_key.strip()
                st.session_state.scene_clips = []
                progress = st.progress(0)

                try:
                    for idx, (img_path, scene) in enumerate(zip(st.session_state.images, st.session_state.scenes)):
                        with st.spinner(f"Animating Scene {idx + 1}..."):
                            image_url = fal_client.upload_file(img_path)
                            result = fal_client.subscribe(
                                i2v_model,
                                arguments={
                                    "image_url": image_url,
                                    "prompt": f"{scene['visual']}. {motion_hint}".strip(),
                                },
                            )
                            video_url = extract_video_url(result)
                            clip_resp = requests.get(video_url, timeout=120)
                            clip_resp.raise_for_status()

                            clip_path = f"clip_{idx}.mp4"
                            with open(clip_path, "wb") as f:
                                f.write(clip_resp.content)
                            st.session_state.scene_clips.append(clip_path)

                        progress.progress((idx + 1) / len(st.session_state.images))

                    st.success("Moving character clips are ready!")
                except Exception as exc:
                    st.error(f"Image-to-video failed: {exc}")

    if st.session_state.scene_clips:
        for idx, clip_path in enumerate(st.session_state.scene_clips):
            if os.path.exists(clip_path):
                st.video(clip_path)

# --- TAB 3: VOICE & SOUND ---
with tab3:
    st.subheader("Select Narrator Voice")
    voice_map = {
        "Warm Male Storyteller (US)": "en-US-ChristopherNeural",
        "Expressive Female Storyteller (US)": "en-US-AriaNeural",
        "Indian Male Narrator": "en-IN-PrabhatNeural",
        "Indian Female Narrator": "en-IN-NeerjaNeural",
        "British Documentary Male": "en-GB-RyanNeural",
    }
    selected_voice = st.selectbox("Voice Options", list(voice_map.keys()))

    async def make_audio(text, voice, filename):
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(filename)

    if st.button("🎙️ Preview Sample Voice"):
        try:
            with st.spinner("Generating voice sample..."):
                run_async(
                    make_audio(
                        "Once upon a time, a small spark changed everything.",
                        voice_map[selected_voice],
                        "sample.mp3",
                    )
                )
            st.audio("sample.mp3")
        except Exception as exc:
            st.error(f"Could not generate sample audio: {exc}")

    if st.button("🔊 Generate Full Story Narration", type="primary"):
        if not st.session_state.scenes:
            st.warning("Generate your script in Tab 1 first.")
        else:
            try:
                st.session_state.audio_files = []
                progress = st.progress(0)
                for idx, scene in enumerate(st.session_state.scenes):
                    narration_text = scene["narration"].strip() or f"Scene {idx + 1}."
                    aud_path = f"voice_{idx}.mp3"
                    run_async(make_audio(narration_text, voice_map[selected_voice], aud_path))
                    st.session_state.audio_files.append(aud_path)
                    progress.progress((idx + 1) / len(st.session_state.scenes))
                st.success("All voiceovers recorded! Go to Tab 4 to render.")
            except Exception as exc:
                st.error(f"Error generating narration audio: {exc}")

# --- TAB 4: RENDER & DOWNLOAD ---
with tab4:
    st.subheader("Assemble & Download Final Movie")
    if st.button("🎬 Render Final MP4 Video", type="primary"):
        if not st.session_state.images or not st.session_state.audio_files:
            st.error("Make sure both Scene Artwork (Tab 2) and Narration (Tab 3) are generated.")
        else:
            clips = []
            audio_clips = []
            final_video = None
            output_file = "animated_story.mp4"

            try:
                with st.spinner("Stitching scenes and exporting MP4..."):
                    use_moving_clips = len(st.session_state.scene_clips) == len(st.session_state.audio_files)

                    for i, aud_p in enumerate(st.session_state.audio_files):
                        audio_clip = AudioFileClip(aud_p)
                        audio_clips.append(audio_clip)
                        target_duration = max(audio_clip.duration + 0.3, 1.0)

                        if use_moving_clips and os.path.exists(st.session_state.scene_clips[i]):
                            vid = VideoFileClip(st.session_state.scene_clips[i])
                            if vid.duration < target_duration:
                                loops = int(target_duration // vid.duration) + 1
                                vid = concatenate_videoclips([vid] * loops)
                            trimmed = apply_subclip(vid, 0, target_duration)
                            scene_clip = apply_audio(trimmed, audio_clip)
                        else:
                            fitted_img = fit_image_to_frame(st.session_state.images[i], width, height)
                            img_c = ImageClip(fitted_img)
                            img_c = apply_duration(img_c, target_duration)
                            scene_clip = apply_audio(img_c, audio_clip)
                        clips.append(scene_clip)

                    final_video = concatenate_videoclips(clips, method="compose")
                    final_video.write_videofile(
                        output_file,
                        fps=24,
                        codec="libx264",
                        audio_codec="aac",
                        logger=None,
                    )

                st.video(output_file)
                with open(output_file, "rb") as f:
                    st.download_button(
                        label="⬇️ Download Video for Reels / YouTube",
                        data=f.read(),
                        file_name="my_animated_story.mp4",
                        mime="video/mp4",
                    )
            except Exception as exc:
                st.error(f"Video rendering failed: {exc}")
            finally:
                for c in clips:
                    try:
                        c.close()
                    except Exception:
                        pass
                for a in audio_clips:
                    try:
                        a.close()
                    except Exception:
                        pass
                if final_video is not None:
                    try:
                        final_video.close()
                    except Exception:
                        pass

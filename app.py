import asyncio
import json
import os
import tempfile
import time
from io import BytesIO

import edge_tts
import fal_client
import numpy as np
import requests
import streamlit as st
from google import genai
from huggingface_hub import InferenceClient
from PIL import Image, ImageDraw

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


def build_motion_clip(img_path, target_w, target_h, duration, motion_mode="Auto Alternate", scene_idx=0):
    """Create a smooth zoom/pan video clip from a still image using Pillow + NumPy."""
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    modes = ["Zoom In", "Pan Right", "Zoom Out", "Pan Left"]
    active_mode = modes[scene_idx % len(modes)] if motion_mode == "Auto Alternate" else motion_mode

    overscan = 1.16
    canvas_w = int(round(target_w * overscan))
    canvas_h = int(round(target_h * overscan))

    with Image.open(img_path) as im:
        rgb = im.convert("RGB")
        cover_scale = max(canvas_w / max(1, rgb.width), canvas_h / max(1, rgb.height))
        base_w = max(canvas_w, int(round(rgb.width * cover_scale)))
        base_h = max(canvas_h, int(round(rgb.height * cover_scale)))
        base_resized = rgb.resize((base_w, base_h), resampling)
        left_pad = max(0, (base_w - canvas_w) // 2)
        top_pad = max(0, (base_h - canvas_h) // 2)
        canvas_np = np.asarray(
            base_resized.crop((left_pad, top_pad, left_pad + canvas_w, top_pad + canvas_h)),
            dtype=np.uint8,
        )

    safe_dur = max(float(duration), 0.1)

    def make_frame(t):
        progress = min(max(float(t) / safe_dur, 0.0), 1.0)
        ease = progress * progress * (3.0 - 2.0 * progress)

        if active_mode == "Zoom In":
            scale = 1.0 + 0.13 * ease
            win_w = max(2, int(round(target_w / scale)))
            win_h = max(2, int(round(target_h / scale)))
            x0 = (canvas_w - win_w) // 2
            y0 = (canvas_h - win_h) // 2
        elif active_mode == "Zoom Out":
            scale = 1.13 - 0.13 * ease
            win_w = max(2, int(round(target_w / scale)))
            win_h = max(2, int(round(target_h / scale)))
            x0 = (canvas_w - win_w) // 2
            y0 = (canvas_h - win_h) // 2
        elif active_mode == "Pan Right":
            win_w = target_w
            win_h = target_h
            max_x = max(0, canvas_w - win_w)
            x0 = int(round(max_x * ease))
            y0 = max(0, (canvas_h - win_h) // 2)
        else:
            win_w = target_w
            win_h = target_h
            max_x = max(0, canvas_w - win_w)
            x0 = int(round(max_x * (1.0 - ease)))
            y0 = max(0, (canvas_h - win_h) // 2)

        x0 = min(max(0, x0), max(0, canvas_w - win_w))
        y0 = min(max(0, y0), max(0, canvas_h - win_h))
        cropped = canvas_np[y0 : y0 + win_h, x0 : x0 + win_w]

        if win_w != target_w or win_h != target_h:
            frame_pil = Image.fromarray(cropped).resize((target_w, target_h), resampling)
            return np.asarray(frame_pil, dtype=np.uint8)
        return cropped

    clip = ImageClip(make_frame)
    return apply_duration(clip, duration)


def make_scene_fallback_card(scene_idx, prompt_text, target_w, target_h):
    """Create a clean gradient storyboard card if an external model times out."""
    img = Image.new("RGB", (target_w, target_h), color=(24, 34, 54))
    draw = ImageDraw.Draw(img)
    draw.rectangle([24, 24, target_w - 24, target_h - 24], outline=(245, 185, 66), width=4)
    draw.text((48, 60), f"Scene {scene_idx + 1} Storyboard", fill=(255, 220, 120))
    wrapped = [prompt_text[i : i + 42] for i in range(0, min(len(prompt_text), 252), 42)]
    y_pos = 120
    for line in wrapped:
        draw.text((48, y_pos), line, fill=(240, 244, 248))
        y_pos += 28
    return img


def generate_scene_image_hf(hf_token, prompt_text, target_w, target_h, scene_idx=0):
    """Generate an image via Hugging Face InferenceClient with multi-model fallback."""
    client = InferenceClient(token=hf_token.strip())
    hf_models = [
        "black-forest-labs/FLUX.1-schnell",
        "stabilityai/stable-diffusion-xl-base-1.0",
    ]
    last_err = None

    for model_id in hf_models:
        for attempt in range(2):
            try:
                pil_img = client.text_to_image(
                    prompt=prompt_text,
                    model=model_id,
                    width=target_w,
                    height=target_h,
                )
                return pil_img.convert("RGB"), None
            except Exception as exc:
                last_err = exc
                time.sleep(1.5 * (attempt + 1))

    fallback_img = make_scene_fallback_card(scene_idx, prompt_text, target_w, target_h)
    return fallback_img, str(last_err)


st.title("🎬 AI Animated Story Studio")
st.caption("Turn rough story ideas into downloadable animated videos.")

with st.sidebar:
    st.header("🔑 Studio Settings")
    api_key = st.text_input("Gemini API Key", type="password")
    hf_token = st.text_input("Hugging Face Token (free image option)", type="password")
    image_provider = st.selectbox(
        "Free Scene Image Engine",
        [
            "Hugging Face Inference (uses HF Token)",
            "Storyboard Card Fallback (No Token)",
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
        width, height = 720, 1280
    else:
        width, height = 1280, 720

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
        value="4 teenage 12-15 years old, one guy fat, one guy lean and other guy slim. a bear which is very giant and huge bulky one",
    )
    scene_count_fallback = st.slider("Scenes to Generate (if Tab 1 script wasn't created)", 2, 6, 4)

    if st.button("🎨 Generate Scene Artwork", type="primary"):
        if image_provider.startswith("Hugging Face") and not hf_token.strip():
            st.error("Please paste a free Hugging Face Token from huggingface.co/settings/tokens in the sidebar (or choose Storyboard Card Fallback).")
        else:
            active_scenes = st.session_state.scenes
            if not active_scenes:
                default_actions = [
                    "walking together near a lush forest edge beside a quiet village at daytime",
                    "spot a giant huge bulky brown bear emerging from dense forest trees in shock",
                    "two boys quickly climbing tall trees while one boy lies flat and holds his breath on the ground",
                    "the giant bear sniffing near the boy on the ground and walking away into the woods",
                ]
                active_scenes = [
                    {
                        "narration": f"Scene {i + 1} narration.",
                        "visual": default_actions[i % len(default_actions)],
                    }
                    for i in range(scene_count_fallback)
                ]
                st.session_state.scenes = active_scenes

            st.session_state.images = []
            st.session_state.scene_clips = []
            progress = st.progress(0)
            warnings_list = []

            for idx, scene in enumerate(active_scenes):
                full_prompt = (
                    f"{art_style}. Characters: {char_desc}. "
                    f"Scene {idx + 1} action: {scene['visual']}"
                ).strip()

                if image_provider.startswith("Hugging Face"):
                    pil_img, err_msg = generate_scene_image_hf(
                        hf_token, full_prompt, width, height, scene_idx=idx
                    )
                    if err_msg:
                        warnings_list.append(f"Scene {idx + 1} used fallback card: {err_msg}")
                else:
                    pil_img = make_scene_fallback_card(idx, full_prompt, width, height)

                img_path = f"scene_{idx}.png"
                pil_img.save(img_path, format="PNG")
                st.session_state.images.append(img_path)
                progress.progress((idx + 1) / len(active_scenes))

            st.success(f"Generated {len(st.session_state.images)} scene-wise images!")
            for w in warnings_list:
                st.warning(w)

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
    free_motion_style = st.selectbox(
        "Free Camera Motion (for Scene Images)",
        ["Auto Alternate", "Zoom In", "Zoom Out", "Pan Right", "Pan Left"],
    )

    if st.button("🎬 Render Final MP4 Video", type="primary"):
        if not st.session_state.images or not st.session_state.audio_files:
            st.error("Make sure both Scene Artwork (Tab 2) and Narration (Tab 3) are generated.")
        else:
            clips = []
            audio_clips = []
            final_video = None
            output_file = "animated_story.mp4"

            try:
                with st.spinner("Applying camera motion, stitching scenes, and exporting MP4..."):
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
                            motion_clip = build_motion_clip(
                                st.session_state.images[i],
                                width,
                                height,
                                target_duration,
                                motion_mode=free_motion_style,
                                scene_idx=i,
                            )
                            scene_clip = apply_audio(motion_clip, audio_clip)
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

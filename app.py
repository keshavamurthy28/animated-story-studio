import asyncio
import json
import os
import tempfile
import time
from io import BytesIO

import edge_tts
import numpy as np
import requests
import streamlit as st
from google import genai
from google.genai import types
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

try:
    from moviepy.editor import AudioFileClip, ImageClip, concatenate_videoclips
    MOVIEPY_V2 = False
except ModuleNotFoundError:
    from moviepy import AudioFileClip, ImageClip, concatenate_videoclips
    MOVIEPY_V2 = True

st.set_page_config(
    page_title="AI Animated Story Studio",
    page_icon="🎬",
    layout="centered",
)

for key, default in [
    ("scenes", []),
    ("images", []),
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


def apply_duration(clip, duration):
    return clip.with_duration(duration) if MOVIEPY_V2 else clip.set_duration(duration)


def apply_audio(clip, audio):
    return clip.with_audio(audio) if MOVIEPY_V2 else clip.set_audio(audio)


def build_motion_clip(img_path, target_w, target_h, duration, motion_mode="Auto Alternate", scene_idx=0):
    """Create a smooth zoom/pan video clip from a scene image using Pillow + NumPy."""
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


def modify_reference_into_scene(ref_pil, target_w, target_h, scene_idx, art_style):
    """Adapt an uploaded reference image into a distinct scene frame with lighting/mood variation."""
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    base = ref_pil.convert("RGB")
    cover_scale = max(target_w / max(1, base.width), target_h / max(1, base.height))
    rw = max(target_w, int(round(base.width * cover_scale)))
    rh = max(target_h, int(round(base.height * cover_scale)))
    resized = base.resize((rw, rh), resampling)

    # Shift crop focus per scene so each scene has a fresh framing
    shifts_x = [0.5, 0.3, 0.7, 0.45, 0.6, 0.5]
    shifts_y = [0.45, 0.55, 0.4, 0.5, 0.35, 0.6]
    sx = shifts_x[scene_idx % len(shifts_x)]
    sy = shifts_y[scene_idx % len(shifts_y)]
    left = int(round(max(0, rw - target_w) * sx))
    top = int(round(max(0, rh - target_h) * sy))
    framed = resized.crop((left, top, left + target_w, top + target_h))

    # Apply subtle style/color grade per scene
    style_lower = art_style.lower()
    if "ghibli" in style_lower or "watercolor" in style_lower:
        framed = framed.filter(ImageFilter.SMOOTH_MORE)
        framed = ImageEnhance.Color(framed).enhance(1.15)
    elif "pixar" in style_lower or "3d" in style_lower:
        framed = ImageEnhance.Contrast(framed).enhance(1.12)
        framed = ImageEnhance.Color(framed).enhance(1.22)
    else:
        framed = ImageEnhance.Contrast(framed).enhance(1.08)

    # Subtle scene-to-scene brightness/warmth variation
    mood_factors = [1.0, 1.05, 0.95, 1.03, 0.98, 1.06]
    framed = ImageEnhance.Brightness(framed).enhance(mood_factors[scene_idx % len(mood_factors)])

    overlay = Image.new("RGBA", (target_w, target_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.rectangle([0, target_h - 110, target_w, target_h], fill=(12, 16, 24, 150))
    draw.text((28, target_h - 82), f"Scene {scene_idx + 1}", fill=(255, 224, 130, 255))

    combined = Image.alpha_composite(framed.convert("RGBA"), overlay)
    return combined.convert("RGB")


def generate_direct_scene_image(gemini_key, prompt_text, ratio_str, target_w, target_h, scene_idx, ref_pil=None):
    """Generate a scene image directly from description (and optional reference image), with safe visual fallback."""
    if gemini_key and gemini_key.strip():
        try:
            client = genai.Client(api_key=gemini_key.strip())
            contents = [ref_pil.convert("RGB"), prompt_text] if ref_pil is not None else prompt_text
            for model_id in ["gemini-3.1-flash-lite-image", "gemini-3.1-flash-image"]:
                try:
                    response = client.models.generate_content(
                        model=model_id,
                        contents=contents,
                        config=types.GenerateContentConfig(
                            response_modalities=["IMAGE"],
                            image_config=types.ImageConfig(aspect_ratio=ratio_str),
                        ),
                    )
                    for cand in getattr(response, "candidates", None) or []:
                        content = getattr(cand, "content", None)
                        for part in getattr(content, "parts", None) or []:
                            inline = getattr(part, "inline_data", None)
                            if inline and getattr(inline, "data", None):
                                return Image.open(BytesIO(inline.data)).convert("RGB")
                except Exception:
                    continue
        except Exception:
            pass

    # If user uploaded a reference photo/image, modify it directly into a scene frame
    if ref_pil is not None:
        return modify_reference_into_scene(ref_pil, target_w, target_h, scene_idx, prompt_text)

    # Otherwise build a rich scene illustration directly from prompt without requiring any token
    palettes = [
        ((34, 85, 68), (118, 185, 120), (246, 206, 112)),
        ((28, 52, 82), (76, 132, 152), (242, 168, 92)),
        ((64, 46, 86), (138, 102, 156), (250, 198, 128)),
        ((82, 54, 36), (168, 118, 72), (252, 214, 130)),
    ]
    c_top, c_mid, c_accent = palettes[scene_idx % len(palettes)]
    img = Image.new("RGB", (target_w, target_h), c_top)
    draw = ImageDraw.Draw(img)

    # Layered horizon & hills for visual scene composition
    horizon_y = int(target_h * 0.56)
    draw.rectangle([0, horizon_y, target_w, target_h], fill=c_mid)
    draw.ellipse(
        [int(target_w * 0.65) - 90, int(target_h * 0.22) - 90, int(target_w * 0.65) + 90, int(target_h * 0.22) + 90],
        fill=c_accent,
    )
    draw.polygon(
        [(0, horizon_y + 40), (int(target_w * 0.35), horizon_y - 90), (int(target_w * 0.72), horizon_y + 45)],
        fill=tuple(max(0, v - 18) for v in c_mid),
    )
    draw.polygon(
        [(int(target_w * 0.35), horizon_y + 60), (int(target_w * 0.78), horizon_y - 70), (target_w, horizon_y + 65)],
        fill=tuple(max(0, v - 28) for v in c_mid),
    )

    draw.rectangle([28, target_h - 135, target_w - 28, target_h - 28], fill=(16, 22, 32))
    draw.text((48, target_h - 115), f"Scene {scene_idx + 1}", fill=c_accent)
    summary = prompt_text[:78]
    draw.text((48, target_h - 85), summary, fill=(242, 246, 250))
    return img


st.title("🎬 AI Animated Story Studio")
st.caption("Turn rough story ideas and reference photos into downloadable animated videos.")

with st.sidebar:
    st.header("🔑 Studio Settings")
    api_key = st.text_input("Gemini API Key", type="password")
    aspect_ratio = st.selectbox("Video Format", ["9:16 (Reels / Shorts)", "16:9 (YouTube)"])
    if "9:16" in aspect_ratio:
        width, height, ratio_str = 720, 1280, "9:16"
    else:
        width, height, ratio_str = 1280, 720, "16:9"

tab1, tab2, tab3, tab4 = st.tabs([
    "1️⃣ Rough Story",
    "2️⃣ Characters & Scenes",
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

# --- TAB 2: DIRECT SCENE ARTWORK & REFERENCE MODIFICATION ---
with tab2:
    st.subheader("Generate Scene Artwork Directly")
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
        "Describe your characters & scene details:",
        value="4 teenage 12-15 years old, one guy fat, one guy lean and other guy slim. a bear which is very giant and huge bulky one",
    )
    ref_file = st.file_uploader(
        "Optional: Attach your photo or a reference image to modify into scenes",
        type=["png", "jpg", "jpeg", "webp"],
    )
    scene_count_fallback = st.slider("Number of Scenes", 2, 6, 4)

    ref_pil = None
    if ref_file is not None:
        ref_pil = Image.open(ref_file).convert("RGB")
        st.image(ref_pil, caption="Uploaded Reference Image", use_container_width=True)

    if st.button("🎨 Generate Scene Artwork", type="primary"):
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
        progress = st.progress(0)

        for idx, scene in enumerate(active_scenes):
            full_prompt = (
                f"{art_style}. Characters: {char_desc}. "
                f"Scene {idx + 1}: {scene['visual']}"
            ).strip()

            pil_img = generate_direct_scene_image(
                gemini_key=api_key,
                prompt_text=full_prompt,
                ratio_str=ratio_str,
                target_w=width,
                target_h=height,
                scene_idx=idx,
                ref_pil=ref_pil,
            )
            img_path = f"scene_{idx}.png"
            pil_img.save(img_path, format="PNG")
            st.session_state.images.append(img_path)
            progress.progress((idx + 1) / len(active_scenes))

        st.success(f"Generated {len(st.session_state.images)} scene images!")

    if st.session_state.images:
        cols = st.columns(2)
        for idx, img_path in enumerate(st.session_state.images):
            if os.path.exists(img_path):
                cols[idx % 2].image(img_path, caption=f"Scene {idx + 1}", use_container_width=True)

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
        "Camera Motion",
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
                    for i, aud_p in enumerate(st.session_state.audio_files):
                        audio_clip = AudioFileClip(aud_p)
                        audio_clips.append(audio_clip)
                        target_duration = max(audio_clip.duration + 0.3, 1.0)

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

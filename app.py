import asyncio
import json
import os
import tempfile
import time
from io import BytesIO

import edge_tts
import numpy as np
import streamlit as st
from google import genai
from google.genai import types
from huggingface_hub import InferenceClient
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
    """Adapt an uploaded reference image into a playful toy-styled scene frame."""
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    base = ref_pil.convert("RGB")
    cover_scale = max(target_w / max(1, base.width), target_h / max(1, base.height))
    rw = max(target_w, int(round(base.width * cover_scale)))
    rh = max(target_h, int(round(base.height * cover_scale)))
    resized = base.resize((rw, rh), resampling)

    shifts_x = [0.5, 0.25, 0.75, 0.45, 0.65, 0.35]
    shifts_y = [0.45, 0.55, 0.4, 0.5, 0.35, 0.6]
    sx = shifts_x[scene_idx % len(shifts_x)]
    sy = shifts_y[scene_idx % len(shifts_y)]
    left = int(round(max(0, rw - target_w) * sx))
    top = int(round(max(0, rh - target_h) * sy))
    framed = resized.crop((left, top, left + target_w, top + target_h))

    framed = framed.filter(ImageFilter.SMOOTH_MORE)
    framed = ImageEnhance.Contrast(framed).enhance(1.14)
    framed = ImageEnhance.Color(framed).enhance(1.28)
    mood_factors = [1.02, 1.06, 0.98, 1.04, 1.0, 1.06]
    framed = ImageEnhance.Brightness(framed).enhance(mood_factors[scene_idx % len(mood_factors)])
    return framed.convert("RGB")


def make_local_scene_card(target_w, target_h, scene_idx, prompt_text):
    """Toy-like rounded character & playful forest fallback when external image APIs are unavailable."""
    top_sky = np.array([92, 198, 245], dtype=np.float32)
    bot_sky = np.array([255, 236, 168], dtype=np.float32)
    if scene_idx % 2 == 1:
        top_sky = np.array([76, 162, 196], dtype=np.float32)
        bot_sky = np.array([255, 214, 148], dtype=np.float32)

    y_idx = np.linspace(0.0, 1.0, target_h)[:, None, None]
    grad = np.broadcast_to(top_sky * (1.0 - y_idx) + bot_sky * y_idx, (target_h, target_w, 3)).astype(np.uint8)
    img = Image.fromarray(grad, mode="RGB")
    draw = ImageDraw.Draw(img, "RGBA")

    ground_y = int(target_h * 0.72)

    # Toy sun with soft halo
    sun_x, sun_y = int(target_w * 0.50), int(target_h * 0.28)
    for rad in range(150, 25, -25):
        draw.ellipse([sun_x - rad, sun_y - rad, sun_x + rad, sun_y + rad], fill=(255, 246, 185, 55))
    draw.ellipse([sun_x - 42, sun_y - 42, sun_x + 42, sun_y + 42], fill=(255, 214, 72, 255), outline=(215, 158, 28, 255), width=3)

    # Toy forest trees (rounded canopies + chunky trunks)
    step = max(85, target_w // 6)
    for idx_t, tx in enumerate(range(35, target_w + 50, step)):
        draw.rounded_rectangle([tx - 11, ground_y - 135, tx + 11, ground_y + 8], radius=6, fill=(138, 86, 46, 255), outline=(68, 38, 18, 255), width=2)
        canopy_col = (96, 204, 92, 255) if idx_t % 2 == 0 else (76, 182, 84, 255)
        draw.ellipse([tx - 44, ground_y - 195, tx + 44, ground_y - 115], fill=canopy_col, outline=(42, 118, 48, 255), width=3)
        draw.ellipse([tx - 24, ground_y - 212, tx + 24, ground_y - 164], fill=canopy_col, outline=(42, 118, 48, 255), width=2)

    # Playful toy grass meadow & soil base
    draw.ellipse([-70, ground_y - 22, target_w + 70, ground_y + 52], fill=(124, 214, 86, 255))
    draw.rectangle([0, ground_y + 15, target_w, target_h], fill=(104, 192, 74, 255))
    soil_y = min(target_h - 30, ground_y + int(target_h * 0.14))
    draw.rectangle([0, soil_y, target_w, target_h], fill=(166, 112, 66, 255))

    # 4 chubby toy boys with big round heads, rosy cheeks, and stubby limbs
    toy_boys = [
        (0.17, 22, 92, (82, 188, 96)),
        (0.29, 24, 96, (76, 178, 224)),
        (0.41, 25, 100, (252, 178, 64)),
        (0.54, 34, 94, (242, 108, 134)),
    ]
    for b_idx, (xf, hw, hh, col) in enumerate(toy_boys):
        bx = int(target_w * xf)
        by = ground_y + 6

        if scene_idx == 2 and b_idx in (0, 1):
            by = ground_y - 125 - b_idx * 18
        elif scene_idx == 2 and b_idx == 3:
            # Toy boy lying flat on the grass
            draw.rounded_rectangle([bx - 36, ground_y - 16, bx + 36, ground_y + 8], radius=12, fill=col, outline=(42, 32, 24, 255), width=3)
            draw.ellipse([bx + 26, ground_y - 34, bx + 62, ground_y + 2], fill=(255, 216, 176, 255), outline=(42, 32, 24, 255), width=3)
            draw.ellipse([bx + 44, ground_y - 22, bx + 50, ground_y - 16], fill=(32, 32, 38, 255))
            continue

        # Stubby toy legs & round shoes
        draw.rounded_rectangle([bx - hw + 5, by - 28, bx - 4, by], radius=7, fill=(68, 92, 132, 255), outline=(42, 32, 24, 255), width=2)
        draw.rounded_rectangle([bx + 4, by - 28, bx + hw - 5, by], radius=7, fill=(68, 92, 132, 255), outline=(42, 32, 24, 255), width=2)
        draw.ellipse([bx - hw + 2, by - 10, bx - 2, by + 6], fill=(56, 62, 78, 255))
        draw.ellipse([bx + 2, by - 10, bx + hw - 2, by + 6], fill=(56, 62, 78, 255))

        # Round toy torso
        draw.rounded_rectangle([bx - hw, by - hh + 34, bx + hw, by - 24], radius=18, fill=col, outline=(42, 32, 24, 255), width=3)

        # Big cute toy head, round ears, rosy cheeks, and big eyes
        hr = 25
        hcy = by - hh + 14
        draw.ellipse([bx - hr - 6, hcy - 5, bx - hr + 6, hcy + 8], fill=(255, 216, 176, 255), outline=(42, 32, 24, 255), width=2)
        draw.ellipse([bx + hr - 6, hcy - 5, bx + hr + 6, hcy + 8], fill=(255, 216, 176, 255), outline=(42, 32, 24, 255), width=2)
        draw.ellipse([bx - hr, hcy - hr, bx + hr, hcy + hr], fill=(255, 216, 176, 255), outline=(42, 32, 24, 255), width=3)
        draw.pieslice([bx - hr, hcy - hr - 3, bx + hr, hcy + 4], start=180, end=360, fill=(112, 68, 38, 255))

        # Rosy cheeks & toy eyes
        draw.ellipse([bx - 16, hcy + 2, bx - 9, hcy + 8], fill=(255, 148, 156, 180))
        draw.ellipse([bx + 9, hcy + 2, bx + 16, hcy + 8], fill=(255, 148, 156, 180))
        draw.ellipse([bx - 11, hcy - 5, bx - 3, hcy + 4], fill=(255, 255, 255, 255))
        draw.ellipse([bx + 3, hcy - 5, bx + 11, hcy + 4], fill=(255, 255, 255, 255))
        draw.ellipse([bx - 9, hcy - 3, bx - 5, hcy + 2], fill=(26, 26, 32, 255))
        draw.ellipse([bx + 5, hcy - 3, bx + 9, hcy + 2], fill=(26, 26, 32, 255))

    # Plump toy bear on the right
    bear_x = int(target_w * 0.78)
    bear_y = ground_y + 6
    draw.ellipse([bear_x - 62, bear_y - 104, bear_x + 62, bear_y - 10], fill=(168, 102, 54, 255), outline=(56, 32, 16, 255), width=3)
    draw.rounded_rectangle([bear_x - 50, bear_y - 34, bear_x - 22, bear_y + 6], radius=10, fill=(152, 90, 46, 255), outline=(56, 32, 16, 255), width=2)
    draw.rounded_rectangle([bear_x + 22, bear_y - 34, bear_x + 50, bear_y + 6], radius=10, fill=(152, 90, 46, 255), outline=(56, 32, 16, 255), width=2)

    bh_x, bh_y = bear_x - 24, bear_y - 112
    draw.ellipse([bh_x - 36, bh_y - 24, bh_x - 10, bh_y + 2], fill=(152, 90, 46, 255), outline=(56, 32, 16, 255), width=2)
    draw.ellipse([bh_x + 10, bh_y - 24, bh_x + 36, bh_y + 2], fill=(152, 90, 46, 255), outline=(56, 32, 16, 255), width=2)
    draw.ellipse([bh_x - 34, bh_y - 14, bh_x + 34, bh_y + 42], fill=(178, 110, 58, 255), outline=(56, 32, 16, 255), width=3)
    draw.ellipse([bh_x - 16, bh_y + 8, bh_x + 16, bh_y + 32], fill=(248, 204, 154, 255))
    draw.ellipse([bh_x - 6, bh_y + 10, bh_x + 6, bh_y + 18], fill=(48, 30, 20, 255))
    draw.ellipse([bh_x - 16, bh_y - 2, bh_x - 6, bh_y + 8], fill=(255, 255, 255, 255))
    draw.ellipse([bh_x + 6, bh_y - 2, bh_x + 16, bh_y + 8], fill=(255, 255, 255, 255))
    draw.ellipse([bh_x - 13, bh_y, bh_x - 9, bh_y + 5], fill=(24, 24, 28, 255))
    draw.ellipse([bh_x + 9, bh_y, bh_x + 13, bh_y + 5], fill=(24, 24, 28, 255))

    return img.convert("RGB")


def generate_scene_image_with_ai(
    engine_choice,
    gemini_key,
    hf_token,
    prompt_text,
    ratio_str,
    target_w,
    target_h,
    scene_idx,
    ref_pil=None,
):
    """Generate cute toy-styled scene images via Hugging Face or Gemini, with toy fallback."""
    errors = []
    toy_prompt = f"Cute toy figure cartoon style, chunky rounded plastic/vinyl toy characters, bright playful forest toy set. {prompt_text}"

    if engine_choice == "Hugging Face (FLUX / SDXL)" and hf_token and hf_token.strip():
        client = InferenceClient(token=hf_token.strip())
        for model_id in [
            "black-forest-labs/FLUX.1-schnell",
            "stabilityai/stable-diffusion-xl-base-1.0",
        ]:
            try:
                pil_img = client.text_to_image(
                    prompt=toy_prompt,
                    model=model_id,
                    width=target_w,
                    height=target_h,
                )
                return pil_img.convert("RGB"), model_id, []
            except Exception as exc:
                errors.append(f"{model_id}: {exc}")

    if gemini_key and gemini_key.strip():
        try:
            client = genai.Client(api_key=gemini_key.strip())
            contents = [ref_pil.convert("RGB"), toy_prompt] if ref_pil is not None else toy_prompt
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
                                return Image.open(BytesIO(inline.data)).convert("RGB"), model_id, []
                    errors.append(f"{model_id}: no image returned")
                except Exception as exc:
                    errors.append(f"{model_id}: {exc}")
        except Exception as exc:
            errors.append(f"Gemini client: {exc}")

    if ref_pil is not None:
        return modify_reference_into_scene(ref_pil, target_w, target_h, scene_idx, toy_prompt), "Toy Reference Frame", errors

    return make_local_scene_card(target_w, target_h, scene_idx, toy_prompt), "Toy Scene Fallback", errors


st.title("🎬 AI Animated Story Studio")
st.caption("Generate cute toy-style story scenes, narration, camera motion, and downloadable MP4 videos.")

with st.sidebar:
    st.header("🔑 Studio Settings")
    api_key = st.text_input("Gemini API Key", type="password")
    hf_token = st.text_input("Hugging Face Token (optional)", type="password")
    image_engine = st.selectbox(
        "Scene Image Model",
        [
            "Hugging Face (FLUX / SDXL)",
            "Gemini Image",
        ],
    )
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
            Fix and rewrite this rough story into an engaging {num_scenes}-scene toy-animated short script.
            Return ONLY valid JSON as a list of objects, with no markdown formatting.
            Each object must have:
            - "narration": clear, emotional voiceover text (1-2 sentences)
            - "visual": detailed toy-scene visual action description (rounded toy characters and toy forest set)
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

# --- TAB 2: CHARACTERS & SCENES ---
with tab2:
    st.subheader("Generate Toy-Style Scene Artwork")
    art_style = st.selectbox(
        "Choose Toy Style",
        [
            "Cute 3D toy / vinyl figure cartoon style, bright playful forest toy set, rounded toy characters",
            "Cute 2D toy storybook cartoon style, big round heads, bright pastel & forest toy colors",
            "Playful miniature toy diorama style, colorful toy trees and cute toy figures",
        ],
    )
    char_desc = st.text_input(
        "Describe your toy characters & scene details:",
        value="Four cute toy boys aged 12-15 (chubby toy boy, lean toy boy, two slim toy boys) and a big plump brown toy bear",
    )
    ref_file = st.file_uploader(
        "Optional: Attach your photo or a reference image to modify into toy scenes",
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
                "four toy boys walking together along a bright toy forest path in sunny morning light",
                "a big plump brown toy bear appearing near the toy trees while the toy boys look surprised",
                "two toy boys climbing toy trees while one toy boy lies still on the toy grass",
                "the plump brown toy bear sniffing near the toy boy on the grass before strolling away",
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
        used_sources = []
        all_errors = []

        for idx, scene in enumerate(active_scenes):
            full_prompt = (
                f"{art_style}. Toy characters: {char_desc}. "
                f"Scene {idx + 1}: {scene['visual']}"
            ).strip()

            pil_img, src_name, src_errs = generate_scene_image_with_ai(
                engine_choice=image_engine,
                gemini_key=api_key,
                hf_token=hf_token,
                prompt_text=full_prompt,
                ratio_str=ratio_str,
                target_w=width,
                target_h=height,
                scene_idx=idx,
                ref_pil=ref_pil,
            )
            used_sources.append(src_name)
            if src_errs:
                all_errors.extend([f"Scene {idx + 1} -> {e}" for e in src_errs])

            img_path = f"scene_{idx}.png"
            pil_img.save(img_path, format="PNG")
            st.session_state.images.append(img_path)
            progress.progress((idx + 1) / len(active_scenes))

        st.success(f"Generated {len(st.session_state.images)} toy scenes ({', '.join(set(used_sources))}).")
        if all_errors:
            with st.expander("Why fallback/reference was used (API details)"):
                for err in all_errors:
                    st.write(err)

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

import asyncio
import base64
import json
import math
import os
import tempfile
import textwrap
import time
from io import BytesIO

import edge_tts
import numpy as np
import requests
import streamlit as st
from google import genai
from google.genai import types
from huggingface_hub import InferenceClient
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

try:
    from moviepy.editor import AudioFileClip, ImageClip, concatenate_videoclips
    MOVIEPY_V2 = False
except ModuleNotFoundError:
    from moviepy import AudioFileClip, ImageClip, concatenate_videoclips
    MOVIEPY_V2 = True

st.set_page_config(
    page_title="AI Animated Story Studio",
    page_icon="🎬",
    layout="wide",
)

for key, default in [
    ("scenes", []),
    ("images", []),
    ("audio_files", []),
    ("locked_character_prompt", "Four teenage boys (12-15 years old, one chubby boy, one lean boy, two slim boys) wearing simple casual clothes, and a large brown forest bear"),
    ("lock_character_enabled", True),
    ("kannada_subtitles", {}),
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


def _find_font(font_size, lang="English"):
    """Load a suitable TrueType font for English or Kannada, fetching Noto Sans Kannada if needed."""
    candidates = []
    if lang == "Kannada":
        candidates.extend([
            "fonts/NotoSansKannada-Regular.ttf",
            "NotoSansKannada-Regular.ttf",
            "NotoSansKannada-Bold.ttf",
            "/tmp/NotoSansKannada-Regular.ttf",
            "/usr/share/fonts/truetype/noto/NotoSansKannada-Regular.ttf",
            "/usr/share/fonts/truetype/noto/NotoSansKannada-Bold.ttf",
        ])
        for path in candidates:
            if os.path.exists(path):
                try:
                    return ImageFont.truetype(path, font_size), path
                except Exception:
                    pass

        tmp_font = "/tmp/NotoSansKannada-Regular.ttf"
        font_urls = [
            "https://github.com/google/fonts/raw/main/ofl/notosanskannada/static/NotoSansKannada-Regular.ttf",
            "https://raw.githubusercontent.com/google/fonts/main/ofl/notosanskannada/NotoSansKannada%5Bwght%5D.ttf",
            "https://raw.githubusercontent.com/notofonts/kannada/main/hinted/ttf/NotoSansKannada-Regular.ttf",
        ]
        for url in font_urls:
            try:
                r = requests.get(url, timeout=20)
                if r.ok and len(r.content) > 15000:
                    with open(tmp_font, "wb") as f:
                        f.write(r.content)
                    return ImageFont.truetype(tmp_font, font_size), tmp_font
            except Exception:
                continue

    candidates.extend([
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "arial.ttf",
        "DejaVuSans-Bold.ttf",
    ])
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, font_size), path
            except Exception:
                pass
    return ImageFont.load_default(), "default_bitmap"


def translate_lines_to_kannada(gemini_key, lines):
    """Translate a list of narration lines into Kannada using Gemini."""
    if not gemini_key or not gemini_key.strip() or not lines:
        return lines

    client = genai.Client(api_key=gemini_key.strip())
    prompt = (
        "Translate each string in this JSON array into clear, natural, easy-to-read Kannada. "
        "Return ONLY a valid JSON array of strings in the exact same order, with no markdown.\n"
        + json.dumps(lines, ensure_ascii=False)
    )
    for model_name in ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-3.6-flash"]:
        try:
            resp = client.models.generate_content(model=model_name, contents=prompt)
            raw = (resp.text or "").strip()
            if raw.startswith("```"):
                raw = "\n".join(
                    line for line in raw.splitlines() if not line.strip().startswith("```")
                ).strip()
            parsed = json.loads(raw)
            if isinstance(parsed, list) and len(parsed) == len(lines):
                return [str(item).strip() for item in parsed]
        except Exception:
            continue
    return lines


def compute_audio_envelope(audio_path, num_steps):
    """Compute normalized speech energy (0.0 to 1.0) across steps for lip-sync cadence."""
    if not audio_path or not os.path.exists(audio_path):
        return [0.45 + 0.35 * abs(math.sin(i * 1.3)) for i in range(num_steps)]
    try:
        clip = AudioFileClip(audio_path)
        try:
            dur = max(float(clip.duration or 1.0), 0.2)
            samples = clip.to_soundarray(fps=4000)
            if samples is None or len(samples) == 0:
                raise ValueError("Empty audio samples")
            mono = np.mean(np.abs(samples), axis=1) if samples.ndim > 1 else np.abs(samples)
            peak = float(np.max(mono)) if len(mono) > 0 else 0.0
            if peak <= 1e-6:
                return [0.15] * num_steps

            env = []
            chunk_len = max(1, len(mono) // num_steps)
            for idx in range(num_steps):
                start = idx * chunk_len
                end = len(mono) if idx == num_steps - 1 else min(len(mono), (idx + 1) * chunk_len)
                seg_rms = float(np.sqrt(np.mean(mono[start:end] ** 2))) if end > start else 0.0
                norm_val = min(1.0, max(0.08, (seg_rms / peak) * 1.45))
                env.append(norm_val)
            return env
        finally:
            clip.close()
    except Exception:
        return [0.35 + 0.45 * abs(math.sin(i * 1.2)) for i in range(num_steps)]


def draw_lipsync_indicator(frame_pil, speech_energy, target_w, target_h, enable_lipsync=True):
    """Draw a subtle animated mouth/lip-sync pulse indicator synced to voiceover energy."""
    if not enable_lipsync:
        return frame_pil

    img_rgba = frame_pil.convert("RGBA")
    draw = ImageDraw.Draw(img_rgba)

    mouth_open = max(2, int(round(4 + speech_energy * 14)))
    mouth_w = max(18, int(round(24 + speech_energy * 12)))
    cx = target_w // 2
    cy = int(round(target_h * 0.74))

    # Subtle lower-third speech/lip-sync cue badge
    badge_w = mouth_w + 26
    badge_h = max(26, mouth_open + 16)
    draw.rounded_rectangle(
        [cx - badge_w // 2, cy - badge_h // 2, cx + badge_w // 2, cy + badge_h // 2],
        radius=badge_h // 2,
        fill=(15, 20, 30, 145),
        outline=(255, 210, 120, 180),
        width=1,
    )
    draw.ellipse(
        [cx - mouth_w // 2, cy - mouth_open // 2, cx + mouth_w // 2, cy + mouth_open // 2],
        fill=(220, 85, 95, 225) if speech_energy > 0.22 else (140, 95, 100, 170),
    )
    return img_rgba.convert("RGB")


def draw_subtitle_on_frame(frame_pil, subtitle_text, target_w, target_h, lang="English"):
    """Draw a clean subtitle banner near the bottom of a scene frame."""
    if not subtitle_text or not subtitle_text.strip() or lang == "None":
        return frame_pil

    img_rgba = frame_pil.convert("RGBA")
    draw = ImageDraw.Draw(img_rgba)

    font_size = max(24, int(round(target_w * 0.038)))
    font, _ = _find_font(font_size, lang=lang)

    chars_per_line = max(16, int(target_w / max(11, font_size * 0.62)))
    wrapped_lines = textwrap.wrap(subtitle_text.strip(), width=chars_per_line)[:3]
    if not wrapped_lines:
        return frame_pil

    line_h = font_size + 10
    box_h = len(wrapped_lines) * line_h + 24
    pad_x = max(20, int(target_w * 0.04))
    box_top = max(16, target_h - box_h - max(42, int(target_h * 0.07)))
    box_bot = min(target_h - 16, box_top + box_h)

    draw.rounded_rectangle(
        [pad_x, box_top, target_w - pad_x, box_bot],
        radius=14,
        fill=(10, 14, 22, 205),
    )

    cur_y = box_top + 12
    for line in wrapped_lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        text_w = bbox[2] - bbox[0]
        text_x = max(pad_x + 12, (target_w - text_w) // 2)
        draw.text((text_x + 2, cur_y + 2), line, font=font, fill=(0, 0, 0, 230))
        draw.text((text_x, cur_y), line, font=font, fill=(255, 248, 214, 255))
        cur_y += line_h

    return img_rgba.convert("RGB")


def build_motion_clip(
    img_path,
    target_w,
    target_h,
    duration,
    motion_mode="Auto Alternate",
    scene_idx=0,
    subtitle_text="",
    subtitle_lang="None",
    audio_path=None,
    enable_lipsync=True,
    enable_char_motion=True,
):
    """Create animated scene frames with camera motion, character movement, lip-sync, and subtitles."""
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    modes = ["Zoom In", "Pan Right", "Zoom Out", "Pan Left"]
    active_mode = modes[scene_idx % len(modes)] if motion_mode == "Auto Alternate" else motion_mode

    overscan = 1.18
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

    safe_dur = max(float(duration), 0.5)
    num_steps = max(6, min(20, int(round(safe_dur * 4))))
    step_dur = safe_dur / num_steps
    speech_env = compute_audio_envelope(audio_path, num_steps)
    subclips = []

    os.makedirs("motion_frames", exist_ok=True)
    for s_idx in range(num_steps):
        progress = s_idx / max(1, num_steps - 1)
        ease = progress * progress * (3.0 - 2.0 * progress)

        # Character breathing & micro-sway offset
        char_sway_x = int(round(math.sin(s_idx * 1.1 + scene_idx) * 8)) if enable_char_motion else 0
        char_bob_y = int(round(abs(math.cos(s_idx * 1.4 + scene_idx)) * 7)) if enable_char_motion else 0

        if active_mode == "Zoom In":
            scale = 1.0 + 0.14 * ease
            win_w = max(2, int(round(target_w / scale)))
            win_h = max(2, int(round(target_h / scale)))
            x0 = (canvas_w - win_w) // 2 + char_sway_x
            y0 = (canvas_h - win_h) // 2 + char_bob_y
        elif active_mode == "Zoom Out":
            scale = 1.14 - 0.14 * ease
            win_w = max(2, int(round(target_w / scale)))
            win_h = max(2, int(round(target_h / scale)))
            x0 = (canvas_w - win_w) // 2 + char_sway_x
            y0 = (canvas_h - win_h) // 2 + char_bob_y
        elif active_mode == "Pan Right":
            win_w = target_w
            win_h = target_h
            max_x = max(0, canvas_w - win_w)
            x0 = int(round(max_x * ease)) + char_sway_x
            y0 = max(0, (canvas_h - win_h) // 2) + char_bob_y
        else:
            win_w = target_w
            win_h = target_h
            max_x = max(0, canvas_w - win_w)
            x0 = int(round(max_x * (1.0 - ease))) + char_sway_x
            y0 = max(0, (canvas_h - win_h) // 2) + char_bob_y

        x0 = min(max(0, x0), max(0, canvas_w - win_w))
        y0 = min(max(0, y0), max(0, canvas_h - win_h))
        cropped = canvas_np[y0 : y0 + win_h, x0 : x0 + win_w]

        frame_pil = Image.fromarray(cropped)
        if win_w != target_w or win_h != target_h:
            frame_pil = frame_pil.resize((target_w, target_h), resampling)

        if enable_lipsync:
            frame_pil = draw_lipsync_indicator(
                frame_pil,
                speech_energy=speech_env[s_idx],
                target_w=target_w,
                target_h=target_h,
                enable_lipsync=enable_lipsync,
            )

        if subtitle_lang != "None" and subtitle_text:
            frame_pil = draw_subtitle_on_frame(
                frame_pil,
                subtitle_text=subtitle_text,
                target_w=target_w,
                target_h=target_h,
                lang=subtitle_lang,
            )

        frame_path = os.path.join("motion_frames", f"s{scene_idx}_f{s_idx}.jpg")
        frame_pil.save(frame_path, format="JPEG", quality=92)
        subclips.append(apply_duration(ImageClip(frame_path), step_dur))

    return concatenate_videoclips(subclips, method="chain")


def modify_reference_into_scene(ref_pil, target_w, target_h, scene_idx, art_style):
    """Adapt an uploaded reference photo/image into a distinct scene frame."""
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

    style_lower = (art_style or "").lower()
    if "ghibli" in style_lower or "watercolor" in style_lower:
        framed = framed.filter(ImageFilter.SMOOTH_MORE)
        framed = ImageEnhance.Color(framed).enhance(1.18)
    else:
        framed = ImageEnhance.Contrast(framed).enhance(1.12)
        framed = ImageEnhance.Color(framed).enhance(1.14)

    mood_factors = [1.0, 1.05, 0.96, 1.03, 0.98, 1.05]
    framed = ImageEnhance.Brightness(framed).enhance(mood_factors[scene_idx % len(mood_factors)])
    return framed.convert("RGB")


def generate_cloudflare_flux_image(cf_account_id, cf_api_token, prompt_text, target_w, target_h, seed=42):
    """Call Cloudflare Workers AI FLUX.1 [schnell] REST API cleanly."""
    url = (
        f"https://api.cloudflare.com/client/v4/accounts/{cf_account_id.strip()}"
        "/ai/run/@cf/black-forest-labs/flux-1-schnell"
    )
    headers = {
        "Authorization": f"Bearer {cf_api_token.strip()}",
        "Content-Type": "application/json",
    }
    payload = {
        "prompt": prompt_text,
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=90)
    if not resp.ok:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")

    content_type = resp.headers.get("Content-Type", "")
    if "image/" in content_type:
        return Image.open(BytesIO(resp.content)).convert("RGB")

    data = resp.json()
    result = data.get("result", {}) if isinstance(data, dict) else {}
    b64_img = result.get("image")
    if b64_img:
        raw_bytes = base64.b64decode(b64_img)
        return Image.open(BytesIO(raw_bytes)).convert("RGB")

    raise ValueError(f"Unexpected Cloudflare response: {data}")


def generate_real_ai_scene(
    engine_choice,
    cf_account_id,
    cf_api_token,
    hf_token,
    gemini_key,
    prompt_text,
    ratio_str,
    target_w,
    target_h,
    scene_idx,
    ref_pil=None,
):
    """Generate realistic AI scene images via Cloudflare Workers AI, Hugging Face, or Gemini."""
    errors = []

    if engine_choice == "Cloudflare Workers AI (Free Daily FLUX)":
        if cf_account_id.strip() and cf_api_token.strip():
            try:
                img = generate_cloudflare_flux_image(
                    cf_account_id, cf_api_token, prompt_text, target_w, target_h, seed=100 + scene_idx
                )
                return img, "Cloudflare FLUX.1-schnell", []
            except Exception as exc:
                errors.append(f"Cloudflare Workers AI: {exc}")
        else:
            errors.append("Missing Cloudflare Account ID or API Token in sidebar.")

    if engine_choice == "Hugging Face (FLUX / SDXL)" or (not errors and hf_token.strip()):
        if hf_token.strip():
            client = InferenceClient(token=hf_token.strip())
            for model_id in [
                "black-forest-labs/FLUX.1-schnell",
                "stabilityai/stable-diffusion-xl-base-1.0",
            ]:
                try:
                    img = client.text_to_image(
                        prompt=prompt_text,
                        model=model_id,
                        width=min(target_w, 1024),
                        height=min(target_h, 1024),
                    )
                    return img.convert("RGB"), f"HF {model_id}", []
                except Exception as exc:
                    errors.append(f"{model_id}: {exc}")
        elif engine_choice == "Hugging Face (FLUX / SDXL)":
            errors.append("Missing Hugging Face Token in sidebar.")

    if engine_choice == "Gemini Image" or (gemini_key.strip() and len(errors) > 0):
        if gemini_key.strip():
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
                                    return Image.open(BytesIO(inline.data)).convert("RGB"), f"Gemini {model_id}", []
                        errors.append(f"{model_id}: no image returned")
                    except Exception as exc:
                        errors.append(f"{model_id}: {exc}")
            except Exception as exc:
                errors.append(f"Gemini client: {exc}")

    if ref_pil is not None:
        return (
            modify_reference_into_scene(ref_pil, target_w, target_h, scene_idx, prompt_text),
            "Uploaded Reference Modification",
            errors,
        )

    return None, None, errors


st.title("🎬 AI Animated Story Studio")
st.caption("Locked character prompts, character reference upload, interactive editing area, voiceovers, character motion & lip-sync, and MP4 export.")

with st.sidebar:
    st.header("🔑 Studio Settings")
    api_key = st.text_input("Gemini API Key (for Script Polishing & Kannada)", type="password")
    image_engine = st.selectbox(
        "AI Scene Image Engine",
        [
            "Cloudflare Workers AI (Free Daily FLUX)",
            "Hugging Face (FLUX / SDXL)",
            "Gemini Image",
        ],
    )
    cf_account_id = st.text_input("Cloudflare Account ID", type="password")
    cf_api_token = st.text_input("Cloudflare API Token", type="password")
    hf_token = st.text_input("Hugging Face Token (optional)", type="password")
    aspect_ratio = st.selectbox("Video Format", ["9:16 (Reels / Shorts)", "16:9 (YouTube)"])
    if "9:16" in aspect_ratio:
        width, height, ratio_str = 720, 1280, "9:16"
    else:
        width, height, ratio_str = 1280, 720, "16:9"

    st.divider()
    st.subheader("🔒 Locked Character Identity")
    st.session_state.lock_character_enabled = st.checkbox(
        "Keep character prompt locked across scenes",
        value=st.session_state.lock_character_enabled,
    )
    st.session_state.locked_character_prompt = st.text_area(
        "Locked Character Prompt",
        value=st.session_state.locked_character_prompt,
        height=95,
    )

tab1, tab2, tab3, tab4 = st.tabs([
    "1️⃣ Rough Story & Editor",
    "2️⃣ Characters & Scenes",
    "3️⃣ Voice & Sound",
    "4️⃣ Render & Download",
])

# --- TAB 1: SCRIPT WRITER & INTERACTIVE EDITING AREA ---
with tab1:
    st.subheader("Write Your Rough Idea & Edit Scenes")
    rough_story = st.text_area(
        "Type your rough story in any style (even broken sentences):",
        placeholder="e.g., 4 hostel boys walk near a forest village, a giant bear appears, two climb a tree, one lies still holding breath, and learns who a true friend is.",
        height=120,
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
            char_context = (
                f"\nConsistent character description to keep across scenes: {st.session_state.locked_character_prompt}"
                if st.session_state.lock_character_enabled
                else ""
            )
            prompt = f"""
            Fix and rewrite this rough story into an engaging {num_scenes}-scene story script.{char_context}
            Return ONLY valid JSON as a list of objects, with no markdown formatting.
            Each object must have:
            - "narration": clear, emotional voiceover text (1-2 sentences)
            - "visual": detailed visual action description for that scene (action, setting, lighting, composition)
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
                st.session_state.kannada_subtitles = {}
                st.success("Script ready! Edit any scene below or move to Tab 2.")
            else:
                st.error(f"Could not generate script after retries/fallbacks: {last_exc}")

    if st.session_state.scenes:
        st.markdown("### ✏️ Interactive Scene & Subtitle Editing Area")
        for i, scene in enumerate(st.session_state.scenes):
            with st.expander(f"Scene {i + 1} Editor", expanded=True):
                col_a, col_b = st.columns(2)
                with col_a:
                    scene["narration"] = st.text_area(
                        f"Scene {i + 1} Narration (English / Voiceover)",
                        value=scene["narration"],
                        key=f"nar_{i}",
                    )
                    scene["visual"] = st.text_input(
                        f"Scene {i + 1} Action / Camera Visual",
                        value=scene["visual"],
                        key=f"vis_{i}",
                    )
                with col_b:
                    kn_default = st.session_state.kannada_subtitles.get(i, "")
                    kn_edited = st.text_area(
                        f"Scene {i + 1} Kannada Subtitle Override (optional)",
                        value=kn_default,
                        placeholder="Leave blank to auto-translate narration to Kannada when rendering",
                        key=f"kn_sub_{i}",
                    )
                    st.session_state.kannada_subtitles[i] = kn_edited

# --- TAB 2: CHARACTER REFERENCE & REALISTIC AI SCENES ---
with tab2:
    st.subheader("Character Reference Upload & Scene Generation")
    art_style = st.selectbox(
        "Choose Visual Style",
        [
            "Photorealistic cinematic film still, natural sunlight, lifelike human expressions and real wildlife detail, 35mm lens",
            "High-detail 3D CGI animation, expressive lifelike characters, rich forest lighting and depth",
            "Detailed storybook illustration, expressive characters, lush forest scenery",
        ],
    )

    ref_col1, ref_col2 = st.columns([1.4, 1])
    with ref_col1:
        active_char_prompt = st.text_input(
            "Active Character Prompt (Synced with Locked Character Prompt)",
            value=st.session_state.locked_character_prompt,
        )
        if st.session_state.lock_character_enabled:
            st.session_state.locked_character_prompt = active_char_prompt
    with ref_col2:
        ref_file = st.file_uploader(
            "Character Reference Image (Optional)",
            type=["png", "jpg", "jpeg", "webp"],
        )

    scene_count_fallback = st.slider("Number of Scenes", 2, 6, 4)

    ref_pil = None
    if ref_file is not None:
        ref_pil = Image.open(ref_file).convert("RGB")
        st.image(ref_pil, caption="Uploaded Character Reference Image", width=260)

    if st.button("🎨 Generate Scene Artwork", type="primary"):
        active_scenes = st.session_state.scenes
        if not active_scenes:
            default_actions = [
                "four teenage boys walking together along a forest path beside a village in golden morning light",
                "a large brown bear emerging from dense forest trees while the boys react in shock",
                "two boys climbing tall forest trees while one boy lies still on the ground holding his breath",
                "the large brown bear sniffing near the boy on the ground before walking back into the woods",
            ]
            active_scenes = [
                {
                    "narration": f"Scene {i + 1} narration.",
                    "visual": default_actions[i % len(default_actions)],
                }
                for i in range(scene_count_fallback)
            ]
            st.session_state.scenes = active_scenes

        char_to_use = (
            st.session_state.locked_character_prompt
            if st.session_state.lock_character_enabled
            else active_char_prompt
        )

        st.session_state.images = []
        progress = st.progress(0)
        used_sources = []
        all_errors = []

        for idx, scene in enumerate(active_scenes):
            full_prompt = (
                f"{art_style}. Locked Characters: {char_to_use}. "
                f"Scene {idx + 1}: {scene['visual']}"
            ).strip()

            pil_img, src_name, src_errs = generate_real_ai_scene(
                engine_choice=image_engine,
                cf_account_id=cf_account_id,
                cf_api_token=cf_api_token,
                hf_token=hf_token,
                gemini_key=api_key,
                prompt_text=full_prompt,
                ratio_str=ratio_str,
                target_w=width,
                target_h=height,
                scene_idx=idx,
                ref_pil=ref_pil,
            )
            if src_errs:
                all_errors.extend([f"Scene {idx + 1}: {e}" for e in src_errs])

            if pil_img is not None:
                img_path = f"scene_{idx}.png"
                pil_img.save(img_path, format="PNG")
                st.session_state.images.append(img_path)
                used_sources.append(src_name or "AI")

            progress.progress((idx + 1) / len(active_scenes))

        if st.session_state.images:
            st.success(f"Generated {len(st.session_state.images)} scenes ({', '.join(set(used_sources))}).")
        else:
            st.error(
                "No real AI image was generated because no active image API credentials were provided or the image API returned an error/quota limit."
            )

        if all_errors:
            with st.expander("API / Quota Details"):
                for err in all_errors:
                    st.write(err)

    if st.session_state.images:
        cols = st.columns(2)
        for idx, img_path in enumerate(st.session_state.images):
            if os.path.exists(img_path):
                cols[idx % 2].image(img_path, caption=f"Scene {idx + 1}", use_container_width=True)

# --- TAB 3: VOICE & SOUND ---
with tab3:
    st.subheader("Select Narrator Voice (English, Kannada Male/Female, or Your Own Voice)")
    voice_map = {
        "Warm Male Storyteller (US)": ("en-US-ChristopherNeural", "English"),
        "Expressive Female Storyteller (US)": ("en-US-AriaNeural", "English"),
        "Indian Male Narrator (English)": ("en-IN-PrabhatNeural", "English"),
        "Indian Female Narrator (English)": ("en-IN-NeerjaNeural", "English"),
        "Kannada Male Narrator (ಕನ್ನಡ - Gagan)": ("kn-IN-GaganNeural", "Kannada"),
        "Kannada Female Narrator (ಕನ್ನಡ - Sapna)": ("kn-IN-SapnaNeural", "Kannada"),
        "British Documentary Male": ("en-GB-RyanNeural", "English"),
        "My Own Voice (Upload Recording)": (None, "Custom"),
    }
    selected_voice = st.selectbox("Voice Options", list(voice_map.keys()))
    voice_id, voice_lang = voice_map[selected_voice]

    async def make_audio(text, voice, filename):
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(filename)

    if voice_id:
        if st.button("🎙️ Preview Sample Voice"):
            try:
                with st.spinner("Generating voice sample..."):
                    sample_text = (
                        "ಒಂದು ಕಾಲದಲ್ಲಿ, ಒಂದು ಸಣ್ಣ ಕಿಡಿ ಎಲ್ಲವನ್ನೂ ಬದಲಾಯಿಸಿತು."
                        if voice_lang == "Kannada"
                        else "Once upon a time, a small spark changed everything."
                    )
                    run_async(make_audio(sample_text, voice_id, "sample.mp3"))
                st.audio("sample.mp3")
            except Exception as exc:
                st.error(f"Could not generate sample audio: {exc}")

        if st.button("🔊 Generate Full Story Narration", type="primary"):
            if not st.session_state.scenes:
                st.warning("Generate your script in Tab 1 first.")
            else:
                try:
                    raw_lines = [
                        (s.get("narration") or f"Scene {idx + 1}.").strip()
                        for idx, s in enumerate(st.session_state.scenes)
                    ]
                    spoken_lines = raw_lines
                    if voice_lang == "Kannada":
                        custom_kn = [
                            st.session_state.kannada_subtitles.get(i, "").strip()
                            for i in range(len(raw_lines))
                        ]
                        if all(custom_kn):
                            spoken_lines = custom_kn
                        elif api_key and api_key.strip():
                            with st.spinner("Translating scene narration to Kannada & generating voiceovers..."):
                                translated = translate_lines_to_kannada(api_key, raw_lines)
                                spoken_lines = [
                                    custom_kn[i] if custom_kn[i] else translated[i]
                                    for i in range(len(raw_lines))
                                ]

                    st.session_state.audio_files = []
                    progress = st.progress(0)
                    for idx, line_text in enumerate(spoken_lines):
                        aud_path = f"voice_{idx}.mp3"
                        run_async(make_audio(line_text, voice_id, aud_path))
                        st.session_state.audio_files.append(aud_path)
                        progress.progress((idx + 1) / len(spoken_lines))
                    st.success("All voiceovers recorded! Go to Tab 4 to render.")
                except Exception as exc:
                    st.error(f"Error generating narration audio: {exc}")
    else:
        st.info(
            "Upload your own recorded voice files below (either one audio file for the full story or separate files per scene)."
        )
        uploaded_voices = st.file_uploader(
            "Upload Your Voice Recording (.mp3, .wav, .m4a, .ogg)",
            type=["mp3", "wav", "m4a", "ogg"],
            accept_multiple_files=True,
        )
        if st.button("💾 Use My Uploaded Voice", type="primary"):
            if not uploaded_voices:
                st.error("Please upload at least one voice recording first.")
            else:
                os.makedirs("custom_voice", exist_ok=True)
                saved_paths = []
                for idx, up_file in enumerate(uploaded_voices):
                    ext = os.path.splitext(up_file.name)[1].lower() or ".mp3"
                    out_p = os.path.join("custom_voice", f"my_voice_{idx}{ext}")
                    with open(out_p, "wb") as f:
                        f.write(up_file.getbuffer())
                    saved_paths.append(out_p)

                scene_total = max(len(st.session_state.scenes), len(st.session_state.images), 1)
                if len(saved_paths) == 1 and scene_total > 1:
                    full_clip = AudioFileClip(saved_paths[0])
                    try:
                        total_dur = max(float(full_clip.duration or 4.0), 1.0)
                        seg_dur = total_dur / scene_total
                        split_paths = []
                        for s_i in range(scene_total):
                            start_t = s_i * seg_dur
                            end_t = min(total_dur, (s_i + 1) * seg_dur)
                            seg_clip = (
                                full_clip.subclipped(start_t, end_t)
                                if hasattr(full_clip, "subclipped")
                                else full_clip.subclip(start_t, end_t)
                            )
                            seg_path = os.path.join("custom_voice", f"scene_voice_{s_i}.mp3")
                            seg_clip.write_audiofile(seg_path, logger=None)
                            seg_clip.close()
                            split_paths.append(seg_path)
                        st.session_state.audio_files = split_paths
                    finally:
                        full_clip.close()
                else:
                    st.session_state.audio_files = saved_paths[:scene_total]

                st.success(f"Saved {len(st.session_state.audio_files)} voice track(s)! Go to Tab 4 to render.")

        if st.session_state.audio_files:
            st.audio(st.session_state.audio_files[0])

# --- TAB 4: RENDER & DOWNLOAD ---
with tab4:
    st.subheader("Character Animation, Lip-Sync, Subtitles & MP4 Export")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        free_motion_style = st.selectbox(
            "Camera Motion",
            ["Auto Alternate", "Zoom In", "Zoom Out", "Pan Right", "Pan Left"],
        )
    with c2:
        subtitle_lang = st.selectbox(
            "Subtitles Language",
            ["None", "English", "Kannada"],
            index=2,
        )
    with c3:
        enable_char_motion = st.checkbox("Character Movement", value=True)
    with c4:
        enable_lipsync = st.checkbox("Voice Lip-Sync Cadence", value=True)

    _, active_font_source = _find_font(28, lang=subtitle_lang if subtitle_lang != "None" else "English")
    if subtitle_lang == "Kannada":
        st.caption(f"Kannada font status: `{active_font_source}`")

    if st.button("🎬 Render Final MP4 Video", type="primary"):
        if not st.session_state.images or not st.session_state.audio_files:
            st.error("Make sure both Scene Artwork (Tab 2) and Narration (Tab 3) are generated.")
        else:
            clips = []
            audio_clips = []
            final_video = None
            output_file = "animated_story.mp4"

            try:
                with st.spinner("Animating characters, syncing speech cadence, rendering subtitles, and exporting MP4..."):
                    raw_lines = [
                        (s.get("narration") or f"Scene {idx + 1}").strip()
                        for idx, s in enumerate(st.session_state.scenes)
                    ]
                    while len(raw_lines) < len(st.session_state.audio_files):
                        raw_lines.append(f"Scene {len(raw_lines) + 1}")

                    if subtitle_lang == "Kannada":
                        custom_kn = [
                            st.session_state.kannada_subtitles.get(i, "").strip()
                            for i in range(len(raw_lines))
                        ]
                        if api_key and api_key.strip():
                            translated = translate_lines_to_kannada(api_key, raw_lines)
                            sub_lines = [
                                custom_kn[i] if custom_kn[i] else translated[i]
                                for i in range(len(raw_lines))
                            ]
                        else:
                            sub_lines = [
                                custom_kn[i] if custom_kn[i] else raw_lines[i]
                                for i in range(len(raw_lines))
                            ]
                            if not any(custom_kn):
                                st.warning(
                                    "Add a Gemini API Key in the sidebar or enter Kannada text in Tab 1's Scene Editor for Kannada subtitles."
                                )
                    elif subtitle_lang == "English":
                        sub_lines = raw_lines
                    else:
                        sub_lines = [""] * len(st.session_state.audio_files)

                    for i, aud_p in enumerate(st.session_state.audio_files):
                        if i >= len(st.session_state.images):
                            break
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
                            subtitle_text=sub_lines[i] if i < len(sub_lines) else "",
                            subtitle_lang=subtitle_lang,
                            audio_path=aud_p,
                            enable_lipsync=enable_lipsync,
                            enable_char_motion=enable_char_motion,
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

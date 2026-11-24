import asyncio
import copy
import json
import math
import os
import random
import textwrap
import time
import urllib.parse
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
    from moviepy.editor import (
        AudioFileClip,
        CompositeAudioClip,
        ImageClip,
        VideoFileClip,
        concatenate_videoclips,
    )
    from moviepy.video.VideoClip import VideoClip as DynamicVideoClip

    MOVIEPY_V2 = False
except ModuleNotFoundError:
    from moviepy import (
        AudioFileClip,
        CompositeAudioClip,
        ImageClip,
        VideoFileClip,
        concatenate_videoclips,
    )
    from moviepy.video.VideoClip import VideoClip as DynamicVideoClip

    MOVIEPY_V2 = True


st.set_page_config(
    page_title="Free AI 2D Reels & Story Studio",
    page_icon="🎬",
    layout="wide",
)

st.markdown(
    """
    <style>
    html, body, [class*="css"] { font-size: 15px; }
    .stApp { background-color: #0b0f17; color: #f0f6fc; }
    .studio-header {
        display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between;
        gap: 10px; background: linear-gradient(90deg, #131926 0%, #1d263b 100%);
        border: 1px solid #30363d; border-radius: 14px; padding: 14px 18px; margin-bottom: 14px;
    }
    .studio-pill {
        background: linear-gradient(90deg, #ff2e63, #ff6b35); color: #ffffff;
        font-weight: 700; font-size: 12px; padding: 4px 12px; border-radius: 999px;
    }
    .studio-sub { color: #9da7b3; font-size: 13px; }
    .stButton > button { border-radius: 10px; padding: 0.55rem 0.95rem; font-weight: 600; min-height: 42px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- PRESETS & CONSTANTS ---
FORMAT_PRESETS = {
    "9:16 Reels / Shorts": (720, 1280, "9:16"),
    "4:5 Portrait Feed": (864, 1080, "4:5"),
    "1:1 Square Post": (1080, 1080, "1:1"),
    "16:9 YouTube / Story": (1280, 720, "16:9"),
}

ANIMATION_STYLES = {
    "2D Warm Illustrated Reel (Like Screenshots)": (
        "Warm modern 2D animated illustration style, expressive cartoon characters with clean dark outlines, "
        "soft golden ambient lighting, cozy expressive faces, clean digital art, no text"
    ),
    "3D Pixar / Disney Feature Film": (
        "3D animated feature film style like Pixar and DreamWorks, expressive faces, "
        "volumetric global illumination, rich textures, dynamic camera, no text"
    ),
    "High-Budget Anime": (
        "High-budget theatrical anime animation, vibrant glowing highlights, expressive character acting, "
        "detailed painted backgrounds, dramatic sky, no text"
    ),
}

MOTION_CHOICES = ["Auto Alternate", "Zoom In", "Zoom Out", "Pan Right", "Pan Left"]
ANIMATION_FPS = 24

TTS_VOICES = {
    "English (Male - Warm Storyteller)": "en-US-ChristopherNeural",
    "English (Female - Warm Narrator)": "en-US-AriaNeural",
    "English (Male - Calm Guy)": "en-US-GuyNeural",
    "English India (Male - Prabhat)": "en-IN-PrabhatNeural",
    "English India (Female - Neerja)": "en-IN-NeerjaNeural",
    "Kannada (Male - Gagan)": "kn-IN-GaganNeural",
    "Kannada (Female - Sapna)": "kn-IN-SapnaNeural",
    "Hindi (Male - Madhur)": "hi-IN-MadhurNeural",
    "Hindi (Female - Swara)": "hi-IN-SwaraNeural",
}

DEFAULT_CHAR_PROMPT = (
    "A young man with short dark hair and warm smile and a young woman with dark hair in a messy bun, "
    "wearing casual cozy clothes, consistent character faces and outfits across every scene"
)

DEFAULT_SCENES = [
    {
        "narration": "The happiest people aren't the ones posting every single win the moment it happens.",
        "kannada_narration": "ತಮ್ಮ ಪ್ರತಿಯೊಂದು ಸಂತೋಷವನ್ನೂ ತಕ್ಷಣ ಪೋಸ್ಟ್ ಮಾಡುವವರು ಅತ್ಯಂತ ಸಂತೋಷವಾಗಿರುವುದಿಲ್ಲ.",
        "motion": "Zoom In",
        "action_motion": "Gentle warm hug, soft breathing and subtle smile, slow cinematic push-in",
        "visual": "Young couple hugging warmly on a bed in soft golden morning light, peaceful happy expressions",
    },
    {
        "narration": "Protect your peace and let your growth speak quietly before you share it with the world.",
        "kannada_narration": "ನಿಮ್ಮ ಶಾಂತಿಯನ್ನು ಕಾಪಾಡಿಕೊಳ್ಳಿ, ಜಗತ್ತಿಗೆ ಹೇಳುವ ಮೊದಲು ನಿಮ್ಮ ಬೆಳವಣಿಗೆ ಬೆಳೆಯಲು ಬಿಡಿ.",
        "motion": "Pan Right",
        "action_motion": "Playful expression, gentle head tilt and warm smile, smooth pan across room",
        "visual": "Close-up of the young man smiling warmly with soft golden rim light and cozy background",
    },
    {
        "narration": "Not everyone around you wants to see you win, so keep your plans private.",
        "kannada_narration": "ನಿಮ್ಮ ಗೆಲುವನ್ನು ನೋಡಲು ಎಲ್ಲರೂ ಬಯಸುವುದಿಲ್ಲ, ಆದ್ದರಿಂದ ನಿಮ್ಮ ಯೋಜನೆಗಳನ್ನು ಗುಟ್ಟಾಗಿಡಿ.",
        "motion": "Zoom Out",
        "action_motion": "Playful light-hearted moment, subtle eye movement and warm expressions",
        "visual": "Young woman and young man playfully holding tape over their mouths in a bright room, warm 2D illustration",
    },
    {
        "narration": "Celebrate silently first, and share when your blessings are truly permanent.",
        "kannada_narration": "ಮೊದಲು ಮೌನವಾಗಿ ಸಂಭ್ರಮಿಸಿ, ನಿಮ್ಮ ಸಂತೋಷ ಶಾಶ್ವತವಾದಾಗ ಹಂಚಿಕೊಳ್ಳಿ.",
        "motion": "Pan Left",
        "action_motion": "Joyful smile holding house keys up gently, bright natural window light",
        "visual": "Young couple standing together in a bright sunlit apartment with large windows, man holding keys smiling",
    },
]


def _get_secret(key_name: str) -> str:
    try:
        if key_name in st.secrets:
            return str(st.secrets[key_name] or "")
    except Exception:
        pass
    return os.environ.get(key_name, "")


# --- SESSION STATE ---
defaults = {
    "scenes": copy.deepcopy(DEFAULT_SCENES),
    "images": [],
    "scene_videos": [],
    "ref_photo_path": "",
    "audio_files": [],
    "bgm_audio_path": "",
    "bgm_volume": 0.16,
    "voice_volume": 1.0,
    "locked_character_prompt": DEFAULT_CHAR_PROMPT,
    "lock_character_enabled": True,
    "animation_style_name": "2D Warm Illustrated Reel (Like Screenshots)",
    "format_preset": "9:16 Reels / Shorts",
    "hook_banner_enabled": True,
    "hook_line_1": "Keep Your Blessings Quiet",
    "hook_line_2": "Until They're Permanent",
    "highlight_words": "Quiet, Permanent",
    "subtitle_lang": "English",
    "gen_batch_id": 0,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# --- MOVIEPY COMPATIBILITY HELPERS ---
def set_dur(clip, d):
    return clip.with_duration(d) if MOVIEPY_V2 else clip.set_duration(d)


def set_aud(clip, a):
    return clip.with_audio(a) if MOVIEPY_V2 else clip.set_audio(a)


def vol_scale(clip, factor):
    if clip is None or abs(factor - 1.0) < 1e-3:
        return clip
    if MOVIEPY_V2 and hasattr(clip, "with_volume_scaled"):
        return clip.with_volume_scaled(factor)
    if hasattr(clip, "volumex"):
        return clip.volumex(factor)
    return clip


def loop_clip(clip, target_duration):
    cur = float(clip.duration or 0.1)
    if cur <= 0:
        cur = 0.1
    if target_duration <= cur:
        if MOVIEPY_V2 and hasattr(clip, "subclipped"):
            return set_dur(clip.subclipped(0, target_duration), target_duration)
        return set_dur(clip.subclip(0, target_duration), target_duration)
    if hasattr(clip, "loop"):
        return set_dur(clip.loop(duration=target_duration), target_duration)
    reps = int(math.ceil(target_duration / cur))
    return set_dur(concatenate_videoclips([clip] * reps), target_duration)


def run_async(coro):
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            new_loop = asyncio.new_event_loop()
            try:
                return new_loop.run_until_complete(coro)
            finally:
                new_loop.close()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


# --- IMAGE & FRAME COMPOSITING ---
def fit_image_to_box(pil_img: Image.Image, box_w: int, box_h: int) -> Image.Image:
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    rgb = pil_img.convert("RGB")
    scale = max(box_w / max(1, rgb.width), box_h / max(1, rgb.height))
    nw = max(box_w, int(round(rgb.width * scale)))
    nh = max(box_h, int(round(rgb.height * scale)))
    resized = rgb.resize((nw, nh), resampling)
    left = max(0, (nw - box_w) // 2)
    top = max(0, (nh - box_h) // 2)
    return resized.crop((left, top, left + box_w, top + box_h)).resize((box_w, box_h), resampling)


@st.cache_resource(show_spinner=False)
def load_font(size: int, lang: str = "English"):
    if lang == "Kannada":
        kn_path = "/tmp/NotoSansKannada-Bold.ttf"
        if not os.path.exists(kn_path):
            try:
                r = requests.get(
                    "https://github.com/google/fonts/raw/main/ofl/notosanskannada/static/NotoSansKannada-Bold.ttf",
                    timeout=20,
                )
                if r.ok and len(r.content) > 40000:
                    with open(kn_path, "wb") as f:
                        f.write(r.content)
            except Exception:
                pass
        if os.path.exists(kn_path):
            return ImageFont.truetype(kn_path, int(size))

    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "arialbd.ttf",
    ):
        if os.path.exists(candidate):
            return ImageFont.truetype(candidate, int(size))
    return ImageFont.load_default()


def draw_highlighted_line(draw: ImageDraw.Draw, line: str, y: int, canvas_w: int, font, highlight_set: set):
    words = line.split()
    if not words:
        return
    space_w = draw.textbbox((0, 0), " ", font=font)[2] - draw.textbbox((0, 0), " ", font=font)[0]
    word_boxes = []
    total_w = 0
    for w in words:
        bb = draw.textbbox((0, 0), w, font=font)
        ww = bb[2] - bb[0]
        word_boxes.append((w, ww))
        total_w += ww
    total_w += max(0, len(words) - 1) * space_w

    cur_x = max(16, (canvas_w - total_w) // 2)
    for w, ww in word_boxes:
        clean_word = w.strip(".,!?:;\"'()").lower()
        is_hi = clean_word in highlight_set
        fill_color = (255, 222, 89, 255) if is_hi else (255, 255, 255, 255)
        draw.text((cur_x + 2, y + 2), w, font=font, fill=(0, 0, 0, 230))
        draw.text((cur_x, y), w, font=font, fill=fill_color)
        cur_x += ww + space_w


def compose_reel_frame(
    media_rgb: Image.Image,
    canvas_w: int,
    canvas_h: int,
    hook_enabled: bool,
    hook_l1: str,
    hook_l2: str,
    highlight_words_str: str,
    subtitle_text: str,
    lang: str,
) -> Image.Image:
    """Composes a black-backed Reel frame with optional top hook banner, centered visual, and bottom subtitle."""
    canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 255))
    draw = ImageDraw.Draw(canvas)

    highlight_set = {
        w.strip().lower() for w in (highlight_words_str or "").split(",") if w.strip()
    }

    top_pad = int(canvas_h * 0.14) if hook_enabled and (hook_l1.strip() or hook_l2.strip()) else int(canvas_h * 0.06)
    bot_pad = int(canvas_h * 0.14) if subtitle_text and lang != "None" else int(canvas_h * 0.06)
    media_h = max(240, canvas_h - top_pad - bot_pad)

    fitted = fit_image_to_box(media_rgb, canvas_w, media_h)
    canvas.paste(fitted.convert("RGBA"), (0, top_pad))

    if hook_enabled and (hook_l1.strip() or hook_l2.strip()):
        hook_font_sz = max(22, int(round(canvas_w * 0.044)))
        h_font = load_font(hook_font_sz, lang="English")
        line_step = hook_font_sz + 8
        y_cursor = max(24, int(canvas_h * 0.055))
        for raw_ln in (hook_l1.strip(), hook_l2.strip()):
            if raw_ln:
                for wrapped in textwrap.wrap(raw_ln, width=32)[:2]:
                    draw_highlighted_line(draw, wrapped, y_cursor, canvas_w, h_font, highlight_set)
                    y_cursor += line_step

    if subtitle_text and subtitle_text.strip() and lang != "None":
        sub_sz = max(18, int(round(canvas_w * 0.032)))
        s_font = load_font(sub_sz, lang=lang)
        cpl = max(18, int((canvas_w * 0.86) / max(9, sub_sz * 0.56)))
        sub_lines = textwrap.wrap(subtitle_text.strip(), width=cpl)[:2]
        if sub_lines:
            lh = sub_sz + 7
            box_h = len(sub_lines) * lh + 14
            box_y = min(canvas_h - box_h - 20, top_pad + media_h + 14)
            pad_x = int(canvas_w * 0.05)
            draw.rounded_rectangle(
                [pad_x, box_y, canvas_w - pad_x, box_y + box_h],
                radius=10,
                fill=(18, 22, 32, 215),
            )
            sy = box_y + 7
            for ln in sub_lines:
                bb = draw.textbbox((0, 0), ln, font=s_font)
                tx = max(pad_x + 8, (canvas_w - (bb[2] - bb[0])) // 2)
                draw.text((tx + 1, sy + 1), ln, font=s_font, fill=(0, 0, 0, 255))
                draw.text((tx, sy), ln, font=s_font, fill=(248, 250, 252, 255))
                sy += lh

    return canvas.convert("RGB")


# --- FREE KEYFRAME & VIDEO GENERATION ---
def generate_keyframe_image(
    hf_token: str,
    gemini_key: str,
    prompt_text: str,
    box_w: int,
    box_h: int,
    ref_photo_path: str = "",
    seed: int = 42,
):
    clean_prompt = " ".join((prompt_text or "").split())[:1200]
    errors = []

    # 1. Hugging Face FLUX.1-schnell (free tier with HF token)
    if hf_token and hf_token.strip():
        try:
            client = InferenceClient(token=hf_token.strip())
            w_hf = max(512, min(1024, (box_w // 8) * 8))
            h_hf = max(512, min(1024, (box_h // 8) * 8))
            img = client.text_to_image(
                prompt=clean_prompt,
                model="black-forest-labs/FLUX.1-schnell",
                width=w_hf,
                height=h_hf,
            )
            return fit_image_to_box(img, box_w, box_h), "Hugging Face FLUX.1-schnell", errors
        except Exception as exc:
            errors.append(f"HF Image: {exc}")

    # 2. Gemini Image Generation (with optional reference photo)
    if gemini_key and gemini_key.strip():
        try:
            client = genai.Client(api_key=gemini_key.strip())
            contents = [clean_prompt]
            if ref_photo_path and os.path.exists(ref_photo_path):
                with Image.open(ref_photo_path) as ref_im:
                    contents = [
                        ref_im.convert("RGB"),
                        f"Keep the exact same character appearance and style from this reference image. {clean_prompt}",
                    ]
            resp = client.models.generate_content(
                model="gemini-2.5-flash-image",
                contents=contents,
                config=types.GenerateContentConfig(response_modalities=["IMAGE"]),
            )
            for cand in getattr(resp, "candidates", None) or []:
                for part in getattr(getattr(cand, "content", None), "parts", None) or []:
                    inline = getattr(part, "inline_data", None)
                    if inline and getattr(inline, "data", None):
                        im = Image.open(BytesIO(inline.data))
                        return fit_image_to_box(im, box_w, box_h), "Gemini 2.5 Flash Image", errors
        except Exception as exc:
            errors.append(f"Gemini Image: {exc}")

    # 3. Pollinations free no-key fallback
    try:
        w_p = max(384, min(1024, (box_w // 8) * 8))
        h_p = max(384, min(1024, (box_h // 8) * 8))
        encoded = urllib.parse.quote(clean_prompt[:320], safe="")
        url = f"https://image.pollinations.ai/prompt/{encoded}?width={w_p}&height={h_p}&seed={seed}&model=flux&nologo=true"
        r = requests.get(url, timeout=45)
        if r.ok and "image/" in (r.headers.get("Content-Type") or ""):
            im = Image.open(BytesIO(r.content))
            return fit_image_to_box(im, box_w, box_h), "Pollinations FLUX", errors
        errors.append(f"Pollinations HTTP {r.status_code}")
    except Exception as exc:
        errors.append(f"Pollinations: {exc}")

    # 4. Uploaded reference photo fallback
    if ref_photo_path and os.path.exists(ref_photo_path):
        with Image.open(ref_photo_path) as im:
            return fit_image_to_box(im, box_w, box_h), "Reference Photo", errors

    return None, None, errors


def try_hf_image_to_video(hf_token: str, keyframe_path: str, prompt_text: str, out_mp4: str):
    if not (hf_token and hf_token.strip() and keyframe_path and os.path.exists(keyframe_path)):
        return False, "No HF token or keyframe"
    try:
        client = InferenceClient(token=hf_token.strip())
        if not hasattr(client, "image_to_video"):
            return False, "image_to_video not supported in installed huggingface_hub"
        with open(keyframe_path, "rb") as f:
            img_bytes = f.read()
        for model_id in ["Wan-AI/Wan2.1-I2V-14B-480P", "Lightricks/LTX-Video"]:
            try:
                vid_bytes = client.image_to_video(image=img_bytes, prompt=prompt_text[:800], model=model_id)
                if vid_bytes and len(vid_bytes) > 5000:
                    with open(out_mp4, "wb") as out_f:
                        out_f.write(vid_bytes)
                    return True, f"HF {model_id}"
            except Exception as inner_exc:
                last_err = str(inner_exc)
        return False, last_err
    except Exception as exc:
        return False, str(exc)


def render_smooth_motion_clip(
    img_path: str,
    out_mp4: str,
    box_w: int,
    box_h: int,
    duration: float = 4.0,
    motion_mode: str = "Auto Alternate",
    scene_idx: int = 0,
):
    """Smooth pan/zoom + subtle warm light particles—completely free and works offline/in cloud."""
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    safe_dur = max(1.0, float(duration))
    modes = ["Zoom In", "Pan Right", "Zoom Out", "Pan Left"]
    active = modes[scene_idx % len(modes)] if motion_mode == "Auto Alternate" else motion_mode

    with Image.open(img_path) as im:
        base = fit_image_to_box(im, box_w, box_h)

    overscan = 1.16
    cw, ch = int(box_w * overscan), int(box_h * overscan)
    canvas_img = base.resize((cw, ch), resampling)

    rng = random.Random(700 + scene_idx)
    motes = [
        (rng.uniform(0.05, 0.95), rng.uniform(0.08, 0.95), rng.uniform(0.03, 0.10), rng.randint(2, 4), rng.randint(90, 175))
        for _ in range(16)
    ]

    def make_frame(t):
        u = max(0.0, min(1.0, float(t) / safe_dur))
        ease = u * u * (3.0 - 2.0 * u)

        if active == "Zoom In":
            scale = 1.0 + 0.12 * ease
        elif active == "Zoom Out":
            scale = 1.13 - 0.12 * ease
        else:
            scale = 1.08

        win_w = max(2, int(round(box_w / scale)))
        win_h = max(2, int(round(box_h / scale)))

        if active == "Pan Right":
            x0 = int(round((cw - win_w) * ease))
            y0 = (ch - win_h) // 2 + int(round(math.sin(u * math.pi * 2) * 5))
        elif active == "Pan Left":
            x0 = int(round((cw - win_w) * (1.0 - ease)))
            y0 = (ch - win_h) // 2 + int(round(math.cos(u * math.pi * 2) * 5))
        else:
            x0 = (cw - win_w) // 2 + int(round(math.sin(u * math.pi * 2) * 8))
            y0 = (ch - win_h) // 2 + int(round(math.cos(u * math.pi * 2) * 6))

        x0 = max(0, min(cw - win_w, x0))
        y0 = max(0, min(ch - win_h, y0))
        crop = canvas_img.crop((x0, y0, x0 + win_w, y0 + win_h)).resize((box_w, box_h), resampling)

        overlay = Image.new("RGBA", (box_w, box_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        for mx0, my0, spd, rad, alpha in motes:
            mx = int(((mx0 + math.sin(u * 3.0 + my0 * 5.0) * 0.02) % 1.0) * box_w)
            my = int(((my0 - u * spd) % 1.0) * box_h)
            draw.ellipse([mx - rad, my - rad, mx + rad, my + rad], fill=(255, 224, 150, alpha))

        return np.asarray(Image.alpha_composite(crop.convert("RGBA"), overlay).convert("RGB"), dtype=np.uint8)

    clip = set_dur(DynamicVideoClip(make_frame, duration=safe_dur), safe_dur)
    try:
        clip.write_videofile(
            out_mp4,
            fps=ANIMATION_FPS,
            codec="libx264",
            audio=False,
            ffmpeg_params=["-crf", "20", "-preset", "veryfast", "-pix_fmt", "yuv420p"],
            logger=None,
        )
        return out_mp4
    finally:
        try:
            clip.close()
        except Exception:
            pass


def generate_ai_script_gemini(gemini_key: str, topic: str, num_scenes: int, char_lock: str, style_name: str):
    if not (gemini_key and gemini_key.strip()):
        return None, "Enter a Gemini API key to generate custom scripts, or edit the scenes below directly."
    client = genai.Client(api_key=gemini_key.strip())
    prompt = f"""
Create a {num_scenes}-scene short social media Reel script.
Art Style: {style_name}
Character Consistency Lock: {char_lock}
Topic / Story: {topic}

Return ONLY a valid JSON array of {num_scenes} objects (no markdown fences) with keys:
- "narration": short, punchy English voiceover (1 sentence)
- "kannada_narration": natural Kannada translation (1 sentence)
- "motion": one of ["Zoom In", "Zoom Out", "Pan Right", "Pan Left"]
- "action_motion": subtle character movement & camera motion description
- "visual": scene visual description keeping the exact same characters consistent
"""
    last_err = "Unknown error"
    for model_name in ["gemini-2.5-flash", "gemini-2.0-flash"]:
        try:
            resp = client.models.generate_content(model=model_name, contents=prompt)
            raw = (resp.text or "").strip()
            if raw.startswith("```"):
                raw = "\n".join(ln for ln in raw.splitlines() if not ln.strip().startswith("```")).strip()
            data = json.loads(raw)
            if isinstance(data, list) and data:
                return data[:num_scenes], None
        except Exception as exc:
            last_err = str(exc)
    return None, last_err


# --- HEADER ---
st.markdown(
    """
    <div class="studio-header">
        <div style="display:flex;align-items:center;gap:10px;">
            <span class="studio-pill">FREE REELS STUDIO</span>
            <span style="font-weight:700;font-size:17px;">2D Illustrated Reels & Story Generator</span>
        </div>
        <div class="studio-sub">Hook Banner + Highlighted Words • Character Lock • Free AI Art & Motion • Edge TTS</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- SIDEBAR ---
with st.sidebar:
    st.header("📐 Format & Hook Banner")
    st.session_state.format_preset = st.selectbox(
        "Aspect Ratio",
        list(FORMAT_PRESETS.keys()),
        index=list(FORMAT_PRESETS.keys()).index(st.session_state.format_preset),
    )
    canvas_w, canvas_h, ratio_label = FORMAT_PRESETS[st.session_state.format_preset]
    st.caption(f"Canvas: **{canvas_w} × {canvas_h}** (`{ratio_label}`)")

    st.session_state.hook_banner_enabled = st.checkbox(
        "Show Top Hook Title Banner", value=st.session_state.hook_banner_enabled
    )
    st.session_state.hook_line_1 = st.text_input("Hook Line 1", value=st.session_state.hook_line_1)
    st.session_state.hook_line_2 = st.text_input("Hook Line 2", value=st.session_state.hook_line_2)
    st.session_state.highlight_words = st.text_input(
        "Highlight Words (comma-separated)", value=st.session_state.highlight_words
    )

    st.divider()
    st.subheader("🔑 Optional Free API Keys")
    gemini_key = st.text_input(
        "Gemini API Key (Script / Image)",
        value=_get_secret("GEMINI_API_KEY"),
        type="password",
    )
    hf_token = st.text_input(
        "Hugging Face Token (FLUX / Video)",
        value=_get_secret("HF_TOKEN"),
        type="password",
    )

# --- TABS ---
tab1, tab2, tab3, tab4 = st.tabs([
    "1️⃣ Story & Character",
    "2️⃣ Generate Visuals & Motion",
    "3️⃣ Voiceover & Audio",
    "4️⃣ Export Final Reel",
])

with tab1:
    st.markdown("### 1. Character Lock & Scene Script")
    st.session_state.animation_style_name = st.selectbox(
        "Visual Style",
        list(ANIMATION_STYLES.keys()),
        index=list(ANIMATION_STYLES.keys()).index(st.session_state.animation_style_name),
    )
    st.session_state.lock_character_enabled = st.checkbox(
        "Lock Character Appearance Across All Scenes", value=st.session_state.lock_character_enabled
    )
    st.session_state.locked_character_prompt = st.text_area(
        "Character Description (Keep Identical Across Scenes)",
        value=st.session_state.locked_character_prompt,
        height=85,
    )

    ref_upload = st.file_uploader(
        "Optional Character Reference Image (PNG / JPG)", type=["png", "jpg", "jpeg", "webp"]
    )
    if ref_upload:
        os.makedirs("studio_assets", exist_ok=True)
        rp = os.path.join("studio_assets", f"ref_{int(time.time())}.png")
        Image.open(ref_upload).convert("RGB").save(rp)
        st.session_state.ref_photo_path = rp
        st.image(rp, caption="Character Reference Saved", width=180)

    st.divider()
    story_idea = st.text_area(
        "Reel Theme / Message (Optional — if using Gemini script generator):",
        value="Keep your blessings quiet until they are permanent; protect your peace.",
        height=70,
    )
    n_sc = st.slider("Number of Scenes", 2, 6, len(st.session_state.scenes))
    if st.button("✨ Generate Script with Gemini", type="primary"):
        with st.spinner("Generating scenes..."):
            generated, err = generate_ai_script_gemini(
                gemini_key,
                story_idea,
                n_sc,
                st.session_state.locked_character_prompt,
                st.session_state.animation_style_name,
            )
            if generated:
                st.session_state.scenes = generated
                st.session_state.gen_batch_id += 1
                st.success("Script updated!")
                st.rerun()
            else:
                st.info(f"Using editable default scenes ({err})")

    for idx, sc in enumerate(st.session_state.scenes):
        with st.expander(f"Scene {idx + 1}", expanded=(idx == 0)):
            c1, c2 = st.columns(2)
            with c1:
                sc["narration"] = st.text_area(
                    "English Narration",
                    value=sc.get("narration", ""),
                    key=f"en_{idx}_b{st.session_state.gen_batch_id}",
                )
                sc["kannada_narration"] = st.text_area(
                    "Kannada Narration",
                    value=sc.get("kannada_narration", ""),
                    key=f"kn_{idx}_b{st.session_state.gen_batch_id}",
                )
            with c2:
                sc["visual"] = st.text_area(
                    "Scene Visual Prompt",
                    value=sc.get("visual", ""),
                    key=f"vis_{idx}_b{st.session_state.gen_batch_id}",
                )
                cur_m = sc.get("motion", "Zoom In")
                m_idx = MOTION_CHOICES.index(cur_m) if cur_m in MOTION_CHOICES else 0
                sc["motion"] = st.selectbox(
                    "Camera Motion",
                    MOTION_CHOICES,
                    index=m_idx,
                    key=f"mot_{idx}_b{st.session_state.gen_batch_id}",
                )

with tab2:
    st.markdown("### 2. Generate Scene Art & Animated Clips")
    use_hf_i2v = st.checkbox(
        "Try Hugging Face Image-to-Video first (falls back automatically to smooth motion)",
        value=False,
    )
    clip_duration = st.slider("Seconds per Scene Clip", 2.5, 6.0, 4.0, 0.5)

    if st.button("🚀 Generate All Scene Clips", type="primary", use_container_width=True):
        os.makedirs("studio_assets/frames", exist_ok=True)
        os.makedirs("studio_assets/clips", exist_ok=True)

        style_txt = ANIMATION_STYLES[st.session_state.animation_style_name]
        char_txt = st.session_state.locked_character_prompt if st.session_state.lock_character_enabled else ""
        top_pad = int(canvas_h * 0.14) if st.session_state.hook_banner_enabled else int(canvas_h * 0.06)
        media_box_h = max(240, canvas_h - top_pad - int(canvas_h * 0.14))

        new_imgs = []
        new_clips = []
        prog = st.progress(0.0)

        for idx, sc in enumerate(st.session_state.scenes):
            prompt = f"{style_txt}. {char_txt}. {sc.get('visual', '')}"
            kf_path = os.path.join("studio_assets/frames", f"scene_{idx}_{int(time.time())}.png")
            clip_path = os.path.join("studio_assets/clips", f"scene_{idx}_{int(time.time())}.mp4")

            pil_im, src_label, errs = generate_keyframe_image(
                hf_token=hf_token,
                gemini_key=gemini_key,
                prompt_text=prompt,
                box_w=canvas_w,
                box_h=media_box_h,
                ref_photo_path=st.session_state.ref_photo_path,
                seed=100 + idx * 17,
            )

            if pil_im is not None:
                pil_im.save(kf_path)
                new_imgs.append(kf_path)

                i2v_done = False
                if use_hf_i2v and hf_token.strip():
                    i2v_done, _ = try_hf_image_to_video(
                        hf_token,
                        kf_path,
                        f"{prompt}. {sc.get('action_motion', '')}",
                        clip_path,
                    )
                if not i2v_done:
                    render_smooth_motion_clip(
                        img_path=kf_path,
                        out_mp4=clip_path,
                        box_w=canvas_w,
                        box_h=media_box_h,
                        duration=clip_duration,
                        motion_mode=sc.get("motion", "Zoom In"),
                        scene_idx=idx,
                    )
                new_clips.append(clip_path)
            else:
                st.warning(f"Scene {idx + 1}: Could not fetch image ({'; '.join(errs[-2:])})")

            prog.progress((idx + 1) / len(st.session_state.scenes))

        st.session_state.images = new_imgs
        st.session_state.scene_videos = new_clips
        if new_clips:
            st.success(f"Generated {len(new_clips)} animated scene clips!")

    if st.session_state.scene_videos:
        st.markdown("#### Scene Clip Previews")
        cols = st.columns(min(4, len(st.session_state.scene_videos)))
        for i, vp in enumerate(st.session_state.scene_videos):
            if os.path.exists(vp):
                with cols[i % len(cols)]:
                    st.caption(f"Scene {i + 1}")
                    st.video(vp)

with tab3:
    st.markdown("### 3. Free Neural Voiceover & Background Audio")
    c_v1, c_v2 = st.columns(2)
    with c_v1:
        voice_label = st.selectbox("Voiceover Speaker", list(TTS_VOICES.keys()))
    with c_v2:
        st.session_state.subtitle_lang = st.selectbox(
            "Subtitles & Voice Language", ["English", "Kannada", "None"]
        )

    if st.button("🎙️ Generate Voiceovers (edge-tts)", type="primary"):
        os.makedirs("studio_assets/audio", exist_ok=True)
        voice_code = TTS_VOICES[voice_label]
        paths = []
        with st.spinner("Generating voiceovers..."):
            for idx, sc in enumerate(st.session_state.scenes):
                line = (
                    sc.get("kannada_narration")
                    if st.session_state.subtitle_lang == "Kannada" and sc.get("kannada_narration")
                    else sc.get("narration", f"Scene {idx + 1}")
                )
                mp3_p = os.path.join("studio_assets/audio", f"vo_{idx}_{int(time.time())}.mp3")
                run_async(edge_tts.Communicate(line, voice_code).save(mp3_p))
                paths.append(mp3_p)
        st.session_state.audio_files = paths
        st.success(f"Generated {len(paths)} voiceover files!")

    bgm_up = st.file_uploader("Upload Background Music (MP3 / WAV)", type=["mp3", "wav", "m4a"])
    if bgm_up:
        os.makedirs("studio_assets/audio", exist_ok=True)
        bp = os.path.join("studio_assets/audio", f"bgm_{int(time.time())}.mp3")
        with open(bp, "wb") as f:
            f.write(bgm_up.getbuffer())
        st.session_state.bgm_audio_path = bp
        st.success("Background music uploaded!")

    st.session_state.bgm_volume = st.slider("Background Music Volume", 0.0, 0.5, st.session_state.bgm_volume, 0.02)

with tab4:
    st.markdown("### 4. Render & Download Final Reel")
    if st.button("🎬 Render Final MP4 Reel", type="primary", use_container_width=True):
        valid_clips = [p for p in st.session_state.scene_videos if os.path.exists(p)]
        if not valid_clips:
            st.error("Generate scene clips in Tab 2 first.")
        else:
            os.makedirs("studio_exports", exist_ok=True)
            out_final = os.path.join("studio_exports", f"reel_{int(time.time())}.mp4")
            scene_final_clips = []
            open_handles = []

            with st.spinner("Compositing hook banner, motion, captions, and audio..."):
                for idx, vp in enumerate(valid_clips):
                    vc = VideoFileClip(vp)
                    open_handles.append(vc)

                    vo_clip = None
                    if idx < len(st.session_state.audio_files) and os.path.exists(st.session_state.audio_files[idx]):
                        vo_clip = AudioFileClip(st.session_state.audio_files[idx])
                        open_handles.append(vo_clip)
                        dur = max(float(vc.duration or 3.5), float(vo_clip.duration or 3.5) + 0.35)
                    else:
                        dur = float(vc.duration or 4.0)

                    looped_vc = loop_clip(vc, dur)
                    sc_obj = st.session_state.scenes[idx] if idx < len(st.session_state.scenes) else {}
                    sub_txt = (
                        sc_obj.get("kannada_narration", "")
                        if st.session_state.subtitle_lang == "Kannada"
                        else sc_obj.get("narration", "")
                    )

                    def transform_frame(get_frame, t, s_txt=sub_txt):
                        raw_arr = get_frame(t)
                        pil_raw = Image.fromarray(raw_arr)
                        composed = compose_reel_frame(
                            media_rgb=pil_raw,
                            canvas_w=canvas_w,
                            canvas_h=canvas_h,
                            hook_enabled=st.session_state.hook_banner_enabled,
                            hook_l1=st.session_state.hook_line_1,
                            hook_l2=st.session_state.hook_line_2,
                            highlight_words_str=st.session_state.highlight_words,
                            subtitle_text=s_txt,
                            lang=st.session_state.subtitle_lang,
                        )
                        return np.asarray(composed, dtype=np.uint8)

                    if hasattr(looped_vc, "transform"):
                        styled_clip = looped_vc.transform(transform_frame)
                    else:
                        styled_clip = looped_vc.fl(transform_frame)

                    if vo_clip is not None:
                        styled_clip = set_aud(styled_clip, vol_scale(vo_clip, st.session_state.voice_volume))
                    scene_final_clips.append(styled_clip)

                full_reel = concatenate_videoclips(scene_final_clips, method="compose")

                if st.session_state.bgm_audio_path and os.path.exists(st.session_state.bgm_audio_path):
                    bgm_raw = AudioFileClip(st.session_state.bgm_audio_path)
                    open_handles.append(bgm_raw)
                    bgm_bed = vol_scale(loop_clip(bgm_raw, float(full_reel.duration or 4.0)), st.session_state.bgm_volume)
                    if full_reel.audio is not None:
                        full_reel = set_aud(full_reel, CompositeAudioClip([full_reel.audio, bgm_bed]))
                    else:
                        full_reel = set_aud(full_reel, bgm_bed)

                full_reel.write_videofile(
                    out_final,
                    fps=ANIMATION_FPS,
                    codec="libx264",
                    audio_codec="aac",
                    ffmpeg_params=["-crf", "19", "-preset", "veryfast", "-pix_fmt", "yuv420p"],
                    logger=None,
                )

                for h in open_handles:
                    try:
                        h.close()
                    except Exception:
                        pass

            st.success("Your Reel is ready!")
            st.video(out_final)
            with open(out_final, "rb") as f:
                st.download_button(
                    "⬇️ Download Final MP4",
                    data=f.read(),
                    file_name=os.path.basename(out_final),
                    mime="video/mp4",
                    use_container_width=True,
                )

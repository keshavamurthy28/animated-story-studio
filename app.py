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
from PIL import Image, ImageDraw, ImageFont

try:
    from moviepy.editor import (
        AudioFileClip,
        CompositeAudioClip,
        VideoFileClip,
        concatenate_videoclips,
    )
    from moviepy.video.VideoClip import VideoClip as DynamicVideoClip

    MOVIEPY_V2 = False
except ModuleNotFoundError:
    from moviepy import (
        AudioFileClip,
        CompositeAudioClip,
        VideoFileClip,
        concatenate_videoclips,
    )
    from moviepy.video.VideoClip import VideoClip as DynamicVideoClip

    MOVIEPY_V2 = True

st.set_page_config(
    page_title="All-in-One AI Reels Studio (Free Edition)",
    page_icon="🎬",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp { background-color: #0b0f17; color: #f0f6fc; }
    .studio-header {
        background: linear-gradient(90deg, #131926 0%, #1d263b 100%);
        border: 1px solid #30363d; border-radius: 14px; padding: 14px 18px; margin-bottom: 14px;
    }
    .studio-pill {
        background: linear-gradient(90deg, #ff2e63, #ff6b35); color: #fff;
        font-weight: 700; font-size: 12px; padding: 4px 12px; border-radius: 999px;
    }
    .stButton > button { border-radius: 10px; font-weight: 600; min-height: 42px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- PRESETS MATCHING ALL 4 STYLES IN YOUR VIDEO ---
REEL_TEMPLATES = {
    "1. @wealth Style (2D Illustrated Couple + Top Hook Banner)": {
        "style_prompt": (
            "Warm modern 2D animated illustration style, expressive cartoon characters with clean dark outlines, "
            "soft golden ambient lighting, cozy expressive faces, clean digital art, no text"
        ),
        "layout_mode": "Black Canvas + Top Hook",
        "banner_bg": (0, 0, 0, 255),
        "banner_text_color": (255, 255, 255, 255),
        "highlight_color": (255, 222, 89, 255),
        "hook_l1": "Keep Your Blessings Quiet",
        "hook_l2": "Until They're Permanent",
        "highlights": "Quiet, Permanent",
        "char_lock": "Young man with short dark hair and young woman with messy bun, cozy casual clothes, consistent faces",
        "scenes": [
            {
                "narration": "The happiest people aren't the ones posting every single win online.",
                "kannada_narration": "ತಮ್ಮ ಪ್ರತಿಯೊಂದು ಗೆಲುವನ್ನೂ ಪೋಸ್ಟ್ ಮಾಡುವವರು ಅತ್ಯಂತ ಸಂತೋಷವಾಗಿರುವುದಿಲ್ಲ.",
                "visual": "Young couple counting stacks of cash together at a wooden table under a warm lamp, happy smiles",
                "motion": "Zoom In",
            },
            {
                "narration": "Protect your privacy and keep your plans quiet while you build.",
                "kannada_narration": "ನಿಮ್ಮ ಯೋಜನೆಗಳನ್ನು ಗುಟ್ಟಾಗಿಡಿ ಮತ್ತು ಮೌನವಾಗಿ ಬೆಳೆಯಿರಿ.",
                "visual": "Young woman and young man playfully holding tape over their mouths with knowing smiles",
                "motion": "Pan Right",
            },
            {
                "narration": "Celebrate silently first, and share only when your blessings are permanent.",
                "kannada_narration": "ಮೊದಲು ಮೌನವಾಗಿ ಸಂಭ್ರಮಿಸಿ, ನಿಮ್ಮ ಸಂತೋಷ ಶಾಶ್ವತವಾದಾಗ ಹಂಚಿಕೊಳ್ಳಿ.",
                "visual": "Young couple standing in a bright sunlit apartment holding up house keys happily",
                "motion": "Zoom Out",
            },
        ],
    },
    "2. @cartoonfun.world Style (2D Comedy Cartoon + Green Banner)": {
        "style_prompt": (
            "Clean 2D flat vector cartoon animation style, bold outlines, bright daytime outdoor background, "
            "expressive funny character faces, vivid colors, no text"
        ),
        "layout_mode": "Color Banner + Center Frame",
        "banner_bg": (34, 197, 94, 255),
        "banner_text_color": (255, 255, 255, 255),
        "highlight_color": (254, 240, 138, 255),
        "hook_l1": "ಮೆಡಿಕಲ್ ಗೆ ಹೋಗ್ಬೇಕು.... 😂🤣",
        "hook_l2": "",
        "highlights": "",
        "char_lock": "Funny 2D cartoon young man wearing purple shirt and yellow pants riding a green scooter, consistent face",
        "scenes": [
            {
                "narration": "When you tell your friend you are just going to the medical store!",
                "kannada_narration": "ಯಾರೋ ನಿನ್ನ ಹತ್ರ ರಿಕ್ವೆಸ್ಟ್ ಮಾಡ್ತಾ ಇದ್ದಾರೆ, ಇಂಟರ್ನೆಟ್ ಮೇಲೆ ವೈರಲ್ ಆಗಿದ್ದೀಯಾ!",
                "visual": "2D cartoon man riding a green scooter on a village road with green trees and hills behind",
                "motion": "Pan Right",
            },
            {
                "narration": "And suddenly the whole internet is watching your trip!",
                "kannada_narration": "ಅಲ್ಲ ಅದು ಒಂದು ವಿಡಿಯೋ ಇದೆ ಅಲ್ವಾ, ಮತ್ತೆ ಇದು ಒರಿಜಿನಲ್ ಅಲ್ವಾ!",
                "visual": "Close up of 2D cartoon man on green scooter looking surprised and laughing with wide eyes",
                "motion": "Zoom In",
            },
        ],
    },
    "3. @neuronvisuals Style (Minimalist 3D Stick-Figure Explainer)": {
        "style_prompt": (
            "Minimalist 3D dark-mode infographic animation style, smooth white 3D mannequin character on dark charcoal "
            "background, glowing green progress counters and dramatic spotlight lighting, no text"
        ),
        "layout_mode": "Full Screen 9:16",
        "banner_bg": (0, 0, 0, 0),
        "banner_text_color": (255, 255, 255, 255),
        "highlight_color": (74, 222, 128, 255),
        "hook_l1": "1 INCOME vs SMART INCOME",
        "hook_l2": "",
        "highlights": "SMART",
        "char_lock": "Smooth white minimalist 3D stick figure mannequin in a dark moody studio with glowing green buckets",
        "scenes": [
            {
                "narration": "When you rely on a single paycheck, your income stops the moment you stop working.",
                "kannada_narration": "ನೀವು ಒಂದೇ ಆದಾಯವನ್ನು ಅವಲಂಬಿಸಿದಾಗ, ಕೆಲಸ ನಿಲ್ಲಿಸಿದ ತಕ್ಷಣ ಆದಾಯವೂ ನಿಲ್ಲುತ್ತದೆ.",
                "visual": "Minimalist white 3D mannequin standing under dark vertical bars dropping coins into a glowing green bucket",
                "motion": "Zoom In",
            },
            {
                "narration": "Build systems that keep growing even while you rest.",
                "kannada_narration": "ನೀವು ವಿಶ್ರಾಂತಿ ಪಡೆಯುವಾಗಲೂ ಬೆಳೆಯುವ ವ್ಯವಸ್ಥೆಯನ್ನು ನಿರ್ಮಿಸಿ.",
                "visual": "Minimalist white 3D mannequin relaxing on a chair while glowing green coins multiply in the bucket",
                "motion": "Zoom Out",
            },
        ],
    },
    "4. @_aivision_studios_ Style (3D Pixar Mythological & Talking Animals)": {
        "style_prompt": (
            "3D Pixar and DreamWorks animated movie style, rich cinematic lighting, expressive 3D character faces, "
            "detailed textures, vibrant colors, 8k render, no text"
        ),
        "layout_mode": "Full Screen 9:16",
        "banner_bg": (0, 0, 0, 0),
        "banner_text_color": (255, 255, 255, 255),
        "highlight_color": (255, 222, 89, 255),
        "hook_l1": "",
        "hook_l2": "",
        "highlights": "",
        "char_lock": "3D Pixar-style blue-skinned Lord Yama with golden crown, mustache, and ornate jewelry",
        "scenes": [
            {
                "narration": "Who dares summon Yama Dharma Raja in the middle of the night?",
                "kannada_narration": "ಯಾರೋ ನಿನ್ನ ಹತ್ರ ರಿಕ್ವೆಸ್ಟ್ ಮಾಡ್ತಾ ಇರೋದು! ಧರ್ಮ ಯಮ ಧರ್ಮ!",
                "visual": "3D Pixar-style Lord Yama with golden crown and blue skin looking sternly at night with glowing stars",
                "motion": "Zoom In",
            },
            {
                "narration": "Even the little green snake on the village cot has something to say!",
                "kannada_narration": "ಯಾವುದು ಇದು ಕರ್ಮ! ಓ ನೀನು ನಿನ್ನ ತಗೊಂಡು ಹೋಗ್ತೀಯಾ!",
                "visual": "Cute expressive 3D Pixar-style green cartoon snake sitting on a woven village cot with checkered pillows",
                "motion": "Zoom Out",
            },
        ],
    },
}

TTS_VOICES = {
    "Kannada (Male - Gagan)": "kn-IN-GaganNeural",
    "Kannada (Female - Sapna)": "kn-IN-SapnaNeural",
    "English (Male - Epic Storyteller)": "en-US-ChristopherNeural",
    "English (Female - Warm Narrator)": "en-US-AriaNeural",
    "English India (Male - Prabhat)": "en-IN-PrabhatNeural",
    "Hindi (Male - Madhur)": "hi-IN-MadhurNeural",
}

ANIMATION_FPS = 24
CANVAS_W, CANVAS_H = 720, 1280  # Standard 9:16 Reels resolution


def _get_secret(key_name: str) -> str:
    try:
        if key_name in st.secrets:
            return str(st.secrets[key_name] or "")
    except Exception:
        pass
    return os.environ.get(key_name, "")


# --- SESSION STATE ---
if "selected_template" not in st.session_state:
    st.session_state.selected_template = list(REEL_TEMPLATES.keys())[0]
if "scenes" not in st.session_state:
    st.session_state.scenes = copy.deepcopy(REEL_TEMPLATES[st.session_state.selected_template]["scenes"])
if "images" not in st.session_state:
    st.session_state.images = []
if "scene_videos" not in st.session_state:
    st.session_state.scene_videos = []
if "audio_files" not in st.session_state:
    st.session_state.audio_files = []
if "bgm_path" not in st.session_state:
    st.session_state.bgm_path = ""
if "batch_id" not in st.session_state:
    st.session_state.batch_id = 0


# --- MOVIEPY HELPERS ---
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
    cur = max(0.1, float(clip.duration or 0.1))
    if target_duration <= cur:
        return set_dur(clip.subclipped(0, target_duration) if MOVIEPY_V2 else clip.subclip(0, target_duration), target_duration)
    if hasattr(clip, "loop"):
        return set_dur(clip.loop(duration=target_duration), target_duration)
    reps = int(math.ceil(target_duration / cur))
    return set_dur(concatenate_videoclips([clip] * reps), target_duration)


def run_async(coro):
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            nl = asyncio.new_event_loop()
            try:
                return nl.run_until_complete(coro)
            finally:
                nl.close()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


# --- FONTS & GRAPHICS ---
@st.cache_resource(show_spinner=False)
def load_font(size: int, is_kannada: bool = False):
    if is_kannada:
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

    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "arialbd.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, int(size))
    return ImageFont.load_default()


def fit_cover(pil_img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    rgb = pil_img.convert("RGB")
    scale = max(target_w / max(1, rgb.width), target_h / max(1, rgb.height))
    nw, nh = max(target_w, int(round(rgb.width * scale))), max(target_h, int(round(rgb.height * scale)))
    resized = rgb.resize((nw, nh), resampling)
    left, top = max(0, (nw - target_w) // 2), max(0, (nh - target_h) // 2)
    return resized.crop((left, top, left + target_w, top + target_h)).resize((target_w, target_h), resampling)


def generate_offline_fallback_art(scene_idx: int, target_w: int, target_h: int, label_text: str) -> Image.Image:
    """Guaranteed fallback artwork if all free external AI APIs are rate-limited."""
    img = Image.new("RGB", (target_w, target_h), (24, 30, 44))
    draw = ImageDraw.Draw(img)
    palettes = [
        ((255, 183, 77), (45, 55, 72)),
        ((74, 222, 128), (20, 35, 30)),
        ((96, 165, 250), (25, 35, 58)),
        ((244, 114, 182), (45, 25, 45)),
    ]
    accent, bg_tone = palettes[scene_idx % len(palettes)]
    for y in range(target_h):
        ratio = y / max(1, target_h)
        r = int(bg_tone[0] * (1 - ratio) + 12 * ratio)
        g = int(bg_tone[1] * (1 - ratio) + 16 * ratio)
        b = int(bg_tone[2] * (1 - ratio) + 26 * ratio)
        draw.line([(0, y), (target_w, y)], fill=(r, g, b))

    cx, cy = target_w // 2, target_h // 2
    draw.ellipse([cx - 170, cy - 170, cx + 170, cy + 170], fill=accent)
    draw.ellipse([cx - 110, cy - 110, cx + 110, cy + 110], fill=(255, 248, 230))
    font = load_font(24, is_kannada=any("\u0c80" <= c <= "\u0cff" for c in label_text))
    lines = textwrap.wrap(label_text or f"Scene {scene_idx + 1}", width=28)[:3]
    ty = cy + 195
    for ln in lines:
        bb = draw.textbbox((0, 0), ln, font=font)
        draw.text(((target_w - (bb[2] - bb[0])) // 2, ty), ln, font=font, fill=(255, 255, 255))
        ty += 34
    return img


# --- MULTI-PROVIDER FREE IMAGE GENERATOR (NEVER FAILS) ---
def generate_scene_image(
    gemini_key: str,
    hf_token: str,
    prompt: str,
    target_w: int,
    target_h: int,
    scene_idx: int,
    fallback_label: str,
):
    clean_p = " ".join((prompt or "").split())[:1000]
    logs = []

    # 1. Gemini Image (Free tier via Google AI Studio)
    if gemini_key and gemini_key.strip():
        try:
            client = genai.Client(api_key=gemini_key.strip())
            resp = client.models.generate_content(
                model="gemini-2.5-flash-image",
                contents=[clean_p],
                config=types.GenerateContentConfig(response_modalities=["IMAGE"]),
            )
            for cand in getattr(resp, "candidates", None) or []:
                for part in getattr(getattr(cand, "content", None), "parts", None) or []:
                    inline = getattr(part, "inline_data", None)
                    if inline and getattr(inline, "data", None):
                        return fit_cover(Image.open(BytesIO(inline.data)), target_w, target_h), "Gemini 2.5 Image", logs
        except Exception as e:
            logs.append(f"Gemini: {e}")

    # 2. Hugging Face FLUX.1-schnell (Free tier with HF token)
    if hf_token and hf_token.strip():
        try:
            client = InferenceClient(token=hf_token.strip())
            w_hf = max(512, min(1024, (target_w // 8) * 8))
            h_hf = max(512, min(1024, (target_h // 8) * 8))
            im = client.text_to_image(prompt=clean_p, model="black-forest-labs/FLUX.1-schnell", width=w_hf, height=h_hf)
            return fit_cover(im, target_w, target_h), "Hugging Face FLUX", logs
        except Exception as e:
            logs.append(f"HF: {e}")

    # 3. Pollinations Free No-Key Endpoint (with retry)
    w_p = max(512, min(1024, (target_w // 8) * 8))
    h_p = max(512, min(1024, (target_h // 8) * 8))
    encoded = urllib.parse.quote(clean_p[:320], safe="")
    for attempt in range(2):
        try:
            seed = 100 + scene_idx * 37 + attempt
            url = f"https://image.pollinations.ai/prompt/{encoded}?width={w_p}&height={h_p}&seed={seed}&model=flux&nologo=true"
            r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=45)
            if r.ok and "image/" in (r.headers.get("Content-Type") or ""):
                return fit_cover(Image.open(BytesIO(r.content)), target_w, target_h), "Pollinations Free FLUX", logs
            logs.append(f"Pollinations HTTP {r.status_code}")
        except Exception as e:
            logs.append(f"Pollinations: {e}")
        time.sleep(1.5)

    # 4. Guaranteed Offline Art Fallback (so video rendering never crashes)
    return generate_offline_fallback_art(scene_idx, target_w, target_h, fallback_label), "Offline Art Fallback", logs


def animate_keyframe_to_mp4(img_path: str, out_mp4: str, box_w: int, box_h: int, duration: float, motion: str, idx: int):
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    safe_dur = max(1.5, float(duration))
    with Image.open(img_path) as im:
        base = fit_cover(im, box_w, box_h)

    overscan = 1.15
    cw, ch = int(box_w * overscan), int(box_h * overscan)
    big_im = base.resize((cw, ch), resampling)

    rng = random.Random(300 + idx)
    particles = [(rng.random(), rng.random(), rng.uniform(0.03, 0.09), rng.randint(2, 4)) for _ in range(14)]

    def make_frame(t):
        u = max(0.0, min(1.0, float(t) / safe_dur))
        ease = u * u * (3.0 - 2.0 * u)
        if motion == "Zoom In":
            sc = 1.0 + 0.12 * ease
        elif motion == "Zoom Out":
            sc = 1.13 - 0.12 * ease
        else:
            sc = 1.08

        ww, wh = max(2, int(round(box_w / sc))), max(2, int(round(box_h / sc)))
        if motion == "Pan Right":
            x0 = int(round((cw - ww) * ease))
            y0 = (ch - wh) // 2
        elif motion == "Pan Left":
            x0 = int(round((cw - ww) * (1.0 - ease)))
            y0 = (ch - wh) // 2
        else:
            x0 = (cw - ww) // 2 + int(round(math.sin(u * math.pi * 2) * 7))
            y0 = (ch - wh) // 2 + int(round(math.cos(u * math.pi * 2) * 5))

        x0 = max(0, min(cw - ww, x0))
        y0 = max(0, min(ch - wh, y0))
        crop = big_im.crop((x0, y0, x0 + ww, y0 + wh)).resize((box_w, box_h), resampling)

        ov = Image.new("RGBA", (box_w, box_h), (0, 0, 0, 0))
        d = ImageDraw.Draw(ov)
        for px0, py0, spd, rad in particles:
            px = int(((px0 + math.sin(u * 3 + py0 * 5) * 0.02) % 1.0) * box_w)
            py = int(((py0 - u * spd) % 1.0) * box_h)
            d.ellipse([px - rad, py - rad, px + rad, py + rad], fill=(255, 230, 160, 120))

        return np.asarray(Image.alpha_composite(crop.convert("RGBA"), ov).convert("RGB"), dtype=np.uint8)

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
    finally:
        try:
            clip.close()
        except Exception:
            pass


def compose_final_frame(
    raw_rgb: Image.Image,
    layout_mode: str,
    banner_bg: tuple,
    highlight_color: tuple,
    hook_l1: str,
    hook_l2: str,
    highlights_csv: str,
    subtitle_text: str,
):
    canvas = Image.new("RGBA", (CANVAS_W, CANVAS_H), (0, 0, 0, 255))
    draw = ImageDraw.Draw(canvas)
    hi_words = {w.strip().lower() for w in (highlights_csv or "").split(",") if w.strip()}

    if layout_mode == "Full Screen 9:16":
        fitted = fit_cover(raw_rgb, CANVAS_W, CANVAS_H)
        canvas.paste(fitted.convert("RGBA"), (0, 0))
        media_bottom = CANVAS_H - 180
    else:
        top_h = int(CANVAS_H * 0.16)
        bot_h = int(CANVAS_H * 0.16)
        mid_h = CANVAS_H - top_h - bot_h
        fitted = fit_cover(raw_rgb, CANVAS_W, mid_h)
        canvas.paste(fitted.convert("RGBA"), (0, top_h))
        media_bottom = top_h + mid_h

        # Draw Top Hook Banner (@wealth black style or @cartoonfun.world green pill style)
        if hook_l1.strip() or hook_l2.strip():
            has_kn = any("\u0c80" <= c <= "\u0cff" for c in (hook_l1 + hook_l2))
            font = load_font(32 if not has_kn else 28, is_kannada=has_kn)
            if banner_bg[0] > 0 or banner_bg[1] > 0 or banner_bg[2] > 0:
                draw.rounded_rectangle([32, 65, CANVAS_W - 32, 145], radius=16, fill=banner_bg)

            y_cur = 78
            for line in (hook_l1.strip(), hook_l2.strip()):
                if not line:
                    continue
                words = line.split()
                sp_w = draw.textbbox((0, 0), " ", font=font)[2] - draw.textbbox((0, 0), " ", font=font)[0]
                widths = [draw.textbbox((0, 0), w, font=font)[2] - draw.textbbox((0, 0), w, font=font)[0] for w in words]
                tot_w = sum(widths) + max(0, len(words) - 1) * sp_w
                x_cur = max(20, (CANVAS_W - tot_w) // 2)
                for w, ww in zip(words, widths):
                    clean_w = w.strip(".,!?:;\"'()").lower()
                    col = highlight_color if clean_w in hi_words else (255, 255, 255, 255)
                    draw.text((x_cur + 2, y_cur + 2), w, font=font, fill=(0, 0, 0, 230))
                    draw.text((x_cur, y_cur), w, font=font, fill=col)
                    x_cur += ww + sp_w
                y_cur += 42

    # Bottom Subtitle Caption
    if subtitle_text and subtitle_text.strip():
        has_kn_sub = any("\u0c80" <= c <= "\u0cff" for c in subtitle_text)
        s_font = load_font(24, is_kannada=has_kn_sub)
        lines = textwrap.wrap(subtitle_text.strip(), width=34)[:2]
        box_h = len(lines) * 32 + 16
        box_y = min(CANVAS_H - box_h - 30, media_bottom + 18)
        draw.rounded_rectangle([36, box_y, CANVAS_W - 36, box_y + box_h], radius=12, fill=(15, 20, 30, 215))
        sy = box_y + 8
        for ln in lines:
            bb = draw.textbbox((0, 0), ln, font=s_font)
            tx = max(44, (CANVAS_W - (bb[2] - bb[0])) // 2)
            draw.text((tx, sy), ln, font=s_font, fill=(255, 255, 255, 255))
            sy += 32

    return canvas.convert("RGB")


# --- UI HEADER ---
st.markdown(
    """
    <div class="studio-header">
        <span class="studio-pill">ALL-IN-ONE VIRAL REELS STUDIO</span>
        <h3 style="margin:8px 0 4px 0;">Create All 4 Viral AI Reel Styles (2D Couple • Kannada Cartoon • 3D Explainer • 3D Mythological)</h3>
        <div style="color:#9da7b3;font-size:13px;">Works 100% in Browser (GitHub + Streamlit) • Built-in Offline Fallback • Free Edge-TTS Kannada & English Voices</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- SIDEBAR ---
with st.sidebar:
    st.header("🎨 1. Choose Viral Reel Style")
    chosen_tpl = st.selectbox(
        "Select Template Style",
        list(REEL_TEMPLATES.keys()),
        index=list(REEL_TEMPLATES.keys()).index(st.session_state.selected_template),
    )
    if chosen_tpl != st.session_state.selected_template:
        st.session_state.selected_template = chosen_tpl
        st.session_state.scenes = copy.deepcopy(REEL_TEMPLATES[chosen_tpl]["scenes"])
        st.session_state.batch_id += 1
        st.rerun()

    tpl_cfg = REEL_TEMPLATES[chosen_tpl]

    st.subheader("🏷️ Top Hook Banner")
    hook_l1 = st.text_input("Hook Line 1", value=tpl_cfg["hook_l1"], key=f"hl1_{st.session_state.batch_id}")
    hook_l2 = st.text_input("Hook Line 2", value=tpl_cfg["hook_l2"], key=f"hl2_{st.session_state.batch_id}")
    highlights_csv = st.text_input(
        "Highlight Words (Yellow)", value=tpl_cfg["highlights"], key=f"hi_{st.session_state.batch_id}"
    )

    st.divider()
    st.subheader("🔑 Free AI API Keys (Optional)")
    st.caption("Leave blank to use Pollinations Free + Built-in Art Engine automatically.")
    gemini_key = st.text_input("Google AI Studio Key (Free Gemini)", value=_get_secret("GEMINI_API_KEY"), type="password")
    hf_token = st.text_input("Hugging Face Token (Free FLUX)", value=_get_secret("HF_TOKEN"), type="password")

# --- TABS ---
t1, t2, t3 = st.tabs(["1️⃣ Edit Scenes & Character", "2️⃣ Generate Animated Clips & Voice", "3️⃣ Render & Download MP4"])

with t1:
    st.markdown("### Customize Your Characters & Scenes")
    char_lock = st.text_area(
        "Character Consistency Prompt (Kept identical across every scene):",
        value=tpl_cfg["char_lock"],
        height=70,
        key=f"cl_{st.session_state.batch_id}",
    )

    for idx, sc in enumerate(st.session_state.scenes):
        with st.expander(f"Scene {idx + 1}", expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                sc["narration"] = st.text_area(
                    "English Voiceover / Caption",
                    value=sc.get("narration", ""),
                    key=f"en_{idx}_{st.session_state.batch_id}",
                )
                sc["kannada_narration"] = st.text_area(
                    "Kannada Voiceover / Caption",
                    value=sc.get("kannada_narration", ""),
                    key=f"kn_{idx}_{st.session_state.batch_id}",
                )
            with c2:
                sc["visual"] = st.text_area(
                    "Visual Scene Prompt",
                    value=sc.get("visual", ""),
                    key=f"vis_{idx}_{st.session_state.batch_id}",
                )
                sc["motion"] = st.selectbox(
                    "Camera Motion",
                    ["Zoom In", "Zoom Out", "Pan Right", "Pan Left"],
                    index=["Zoom In", "Zoom Out", "Pan Right", "Pan Left"].index(sc.get("motion", "Zoom In")),
                    key=f"mot_{idx}_{st.session_state.batch_id}",
                )

with t2:
    st.markdown("### Generate Scene Art, Motion Clips & Neural Voiceover")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        voice_name = st.selectbox("Voiceover Speaker", list(TTS_VOICES.keys()))
    with col_b:
        lang_mode = st.selectbox("Script Language", ["Kannada", "English", "None (Music Only)"])
    with col_c:
        sec_per_scene = st.slider("Seconds per Clip", 2.5, 6.0, 4.0, 0.5)

    if st.button("🚀 Generate All Scenes & Voiceovers", type="primary", use_container_width=True):
        os.makedirs("reels_build/img", exist_ok=True)
        os.makedirs("reels_build/vid", exist_ok=True)
        os.makedirs("reels_build/aud", exist_ok=True)

        box_h = CANVAS_H if tpl_cfg["layout_mode"] == "Full Screen 9:16" else int(CANVAS_H * 0.68)
        new_imgs, new_vids, new_auds = [], [], []
        prog = st.progress(0.0)
        status = st.empty()

        for idx, sc in enumerate(st.session_state.scenes):
            status.info(f"Building Scene {idx + 1}/{len(st.session_state.scenes)}...")
            full_prompt = f"{tpl_cfg['style_prompt']}. {char_lock}. {sc.get('visual', '')}"
            label_txt = sc.get("kannada_narration") if lang_mode == "Kannada" else sc.get("narration", "")

            # 1. Generate Keyframe Image
            im_pil, src_used, _ = generate_scene_image(
                gemini_key=gemini_key,
                hf_token=hf_token,
                prompt=full_prompt,
                target_w=CANVAS_W,
                target_h=box_h,
                scene_idx=idx,
                fallback_label=label_txt,
            )
            im_path = os.path.join("reels_build/img", f"s_{idx}_{int(time.time())}.png")
            im_pil.save(im_path)
            new_imgs.append(im_path)

            # 2. Animate into Smooth Motion Clip
            vid_path = os.path.join("reels_build/vid", f"v_{idx}_{int(time.time())}.mp4")
            animate_keyframe_to_mp4(im_path, vid_path, CANVAS_W, box_h, sec_per_scene, sc.get("motion", "Zoom In"), idx)
            new_vids.append(vid_path)

            # 3. Generate Free Neural Voiceover
            if lang_mode != "None (Music Only)":
                spoken = sc.get("kannada_narration") if lang_mode == "Kannada" else sc.get("narration", "")
                if spoken and spoken.strip():
                    aud_path = os.path.join("reels_build/aud", f"a_{idx}_{int(time.time())}.mp3")
                    run_async(edge_tts.Communicate(spoken.strip(), TTS_VOICES[voice_name]).save(aud_path))
                    new_auds.append(aud_path)
                else:
                    new_auds.append("")
            else:
                new_auds.append("")

            st.toast(f"Scene {idx + 1} ready ({src_used})")
            prog.progress((idx + 1) / len(st.session_state.scenes))

        st.session_state.images = new_imgs
        st.session_state.scene_videos = new_vids
        st.session_state.audio_files = new_auds
        status.success("All scenes and voiceovers generated!")

    if st.session_state.scene_videos:
        cols = st.columns(min(3, len(st.session_state.scene_videos)))
        for i, vp in enumerate(st.session_state.scene_videos):
            if os.path.exists(vp):
                with cols[i % len(cols)]:
                    st.caption(f"Scene {i + 1}")
                    st.video(vp)

with t3:
    st.markdown("### Upload Optional Background Music & Export Final Reel")
    bgm_file = st.file_uploader("Upload Background Music (MP3 / WAV)", type=["mp3", "wav", "m4a"])
    if bgm_file:
        os.makedirs("reels_build/aud", exist_ok=True)
        bp = os.path.join("reels_build/aud", f"bgm_{int(time.time())}.mp3")
        with open(bp, "wb") as f:
            f.write(bgm_file.getbuffer())
        st.session_state.bgm_path = bp
        st.success("Background music loaded!")

    bgm_vol = st.slider("Background Music Volume", 0.0, 0.4, 0.15, 0.02)

    if st.button("🎬 Render Final 9:16 MP4 Reel", type="primary", use_container_width=True):
        vids = [p for p in st.session_state.scene_videos if os.path.exists(p)]
        if not vids:
            st.error("Generate scenes in Tab 2 first.")
        else:
            os.makedirs("reels_exports", exist_ok=True)
            final_out = os.path.join("reels_exports", f"Viral_Reel_{int(time.time())}.mp4")
            final_clips, handles = [], []

            with st.spinner("Compositing hook banner, subtitles, animations, and audio..."):
                for idx, vp in enumerate(vids):
                    vc = VideoFileClip(vp)
                    handles.append(vc)

                    ac = None
                    if idx < len(st.session_state.audio_files) and st.session_state.audio_files[idx] and os.path.exists(st.session_state.audio_files[idx]):
                        ac = AudioFileClip(st.session_state.audio_files[idx])
                        handles.append(ac)
                        dur = max(float(vc.duration or 3.5), float(ac.duration or 3.5) + 0.3)
                    else:
                        dur = float(vc.duration or 4.0)

                    lvc = loop_clip(vc, dur)
                    sc_obj = st.session_state.scenes[idx] if idx < len(st.session_state.scenes) else {}
                    sub_txt = (
                        sc_obj.get("kannada_narration", "")
                        if lang_mode == "Kannada"
                        else (sc_obj.get("narration", "") if lang_mode == "English" else "")
                    )

                    def frame_pipe(gf, t, s_txt=sub_txt):
                        pil_in = Image.fromarray(gf(t))
                        comp = compose_final_frame(
                            raw_rgb=pil_in,
                            layout_mode=tpl_cfg["layout_mode"],
                            banner_bg=tpl_cfg["banner_bg"],
                            highlight_color=tpl_cfg["highlight_color"],
                            hook_l1=hook_l1,
                            hook_l2=hook_l2,
                            highlights_csv=highlights_csv,
                            subtitle_text=s_txt,
                        )
                        return np.asarray(comp, dtype=np.uint8)

                    styled = lvc.transform(frame_pipe) if hasattr(lvc, "transform") else lvc.fl(frame_pipe)
                    if ac is not None:
                        styled = set_aud(styled, ac)
                    final_clips.append(styled)

                full_vid = concatenate_videoclips(final_clips, method="compose")

                if st.session_state.bgm_path and os.path.exists(st.session_state.bgm_path):
                    bgm_raw = AudioFileClip(st.session_state.bgm_path)
                    handles.append(bgm_raw)
                    bgm_bed = vol_scale(loop_clip(bgm_raw, float(full_vid.duration or 4.0)), bgm_vol)
                    if full_vid.audio is not None:
                        full_vid = set_aud(full_vid, CompositeAudioClip([full_vid.audio, bgm_bed]))
                    else:
                        full_vid = set_aud(full_vid, bgm_bed)

                full_vid.write_videofile(
                    final_out,
                    fps=ANIMATION_FPS,
                    codec="libx264",
                    audio_codec="aac",
                    ffmpeg_params=["-crf", "19", "-preset", "veryfast", "-pix_fmt", "yuv420p"],
                    logger=None,
                )
                for h in handles:
                    try:
                        h.close()
                    except Exception:
                        pass

            st.success("Reel rendered successfully!")
            st.video(final_out)
            with open(final_out, "rb") as f:
                st.download_button(
                    "⬇️ Download Final Reel MP4",
                    data=f.read(),
                    file_name=os.path.basename(final_out),
                    mime="video/mp4",
                    use_container_width=True,
                )

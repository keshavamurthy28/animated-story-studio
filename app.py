import asyncio
import base64
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
        CompositeVideoClip,
        ImageClip,
        VideoFileClip,
        concatenate_videoclips,
    )
    MOVIEPY_V2 = False
except ModuleNotFoundError:
    from moviepy import (
        AudioFileClip,
        CompositeAudioClip,
        CompositeVideoClip,
        ImageClip,
        VideoFileClip,
        concatenate_videoclips,
    )
    MOVIEPY_V2 = True

st.set_page_config(
    page_title="AI Cinema & Reels Studio Pro (AI Video Edition)",
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
    .compact-preview-box {
        display: flex; flex-direction: column; align-items: center; justify-content: center;
        background: #131822; border: 1px solid #283142; border-radius: 12px; padding: 10px;
    }
    .stButton > button { border-radius: 10px; padding: 0.55rem 0.95rem; font-weight: 600; min-height: 42px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- PRESETS & CONSTANTS ---
FORMAT_PRESETS = {
    "9:16 Reels / Shorts": (720, 1280, "9:16"),
    "16:9 Movie / YouTube": (1280, 720, "16:9"),
    "1:1 Square Post": (1080, 1080, "1:1"),
    "4:5 Portrait Feed": (1080, 1350, "4:5"),
}

ANIMATION_STYLES = {
    "3D Pixar / Disney Feature Film": (
        "3D animated feature film style like Pixar and DreamWorks, expressive faces, "
        "volumetric global illumination, rich textures, dynamic action camera, 8k render"
    ),
    "Epic 3D Unreal Engine 5 Mythological": (
        "Unreal Engine 5 cinematic 3D animation, hyper-detailed armor and divine aura, "
        "dramatic rim lighting, volumetric fog, particles, epic movie framing"
    ),
    "High-Budget Anime (Ufotable / Makoto Shinkai)": (
        "High-budget theatrical anime animation, vibrant glowing effects, dynamic motion lines, "
        "detailed painted backgrounds, expressive character acting, dramatic sky"
    ),
    "2D Illustrated Storybook Motion": (
        "Rich hand-painted 2D animated film cel style, warm golden highlights, "
        "deep atmospheric parallax layers, expressive fluid character poses"
    ),
}

VIDEO_FX_CHOICES = [
    "None", "Zoom Punch", "Shake", "Flash White", "Glitch RGB", "Vintage Sepia", "Cinematic Vignette"
]
MOTION_CHOICES = ["Auto Alternate", "Zoom In", "Zoom Out", "Pan Right", "Pan Left"]
SUBTITLE_POSITIONS = ["Bottom", "Center", "Top"]
ANIMATION_FPS = 24
MAX_HISTORY_STEPS = 3

TTS_VOICES = {
    "English (Male - Epic Storyteller)": "en-US-ChristopherNeural",
    "English (Female - Warm Narrator)": "en-US-AriaNeural",
    "English India (Male - Clear)": "en-IN-PrabhatNeural",
    "English India (Female - Expressive)": "en-IN-NeerjaNeural",
    "Kannada (Male - Gagan)": "kn-IN-GaganNeural",
    "Kannada (Female - Sapna)": "kn-IN-SapnaNeural",
    "Hindi (Male - Madhur)": "hi-IN-MadhurNeural",
    "Hindi (Female - Swara)": "hi-IN-SwaraNeural",
}

PRESET_AUDIO_EFFECTS = {
    "None": "",
    "⚡ Thunder Strike": "https://assets.mixkit.co/active_storage/sfx/1271/1271-preview.mp3",
    "🔥 Campfire / Energy Aura": "https://assets.mixkit.co/active_storage/sfx/2435/2435-preview.mp3",
    "💨 Wind Gust": "https://assets.mixkit.co/active_storage/sfx/2673/2673-preview.mp3",
    "🌊 Ocean / River Waves": "https://assets.mixkit.co/active_storage/sfx/1188/1188-preview.mp3",
    "🌲 Forest Birds": "https://assets.mixkit.co/active_storage/sfx/2436/2436-preview.mp3",
    "💥 Impact Boom": "https://assets.mixkit.co/active_storage/sfx/2578/2578-preview.mp3",
}

DEFAULT_CHAR_PROMPT = (
    "Mighty Lord Hanuman with golden crown, muscular divine form, glowing mace (Gada), "
    "flowing orange silk dhoti, radiant aura, consistent face and attire across every scene"
)

DEFAULT_SCENES = [
    {
        "narration": "Amid the roaring battlefield of Lanka, Hanuman leaps skyward with unstoppable divine strength.",
        "kannada_narration": "ಲಂಕೆಯ ರಣರಂಗದ ನಡುವೆ ಹನುಮಂತನು ಅಪಾರ ದೈವಿಕ ಶಕ್ತಿಯೊಂದಿಗೆ ಆಕಾಶಕ್ಕೆ ಜಿಗಿಯುತ್ತಾನೆ.",
        "emotion": "Heroic & Fierce",
        "motion": "Zoom In",
        "action_motion": "Hero leaping dynamically through golden clouds with glowing aura and flowing cape, fast camera tracking",
        "suggested_fx": "Zoom Punch",
        "suggested_sfx": "💨 Wind Gust",
        "visual": "Mighty Hanuman leaping high over ancient battlefield with glowing mace, dramatic sunbeams, flying embers",
    },
    {
        "narration": "Crossing the vast ocean in a single bound, waves part beneath his radiant presence.",
        "kannada_narration": "ಒಂದೇ ಜಿಗಿತದಲ್ಲಿ ವಿಶಾಲ ಸಮುದ್ರವನ್ನು ದಾಟುತ್ತಾ, ಅವನ ತೇಜಸ್ಸಿನ ಮುಂದೆ ಅಲೆಗಳು ದಾರಿ ಮಾಡಿಕೊಡುತ್ತವೆ.",
        "emotion": "Majestic & Powerful",
        "motion": "Pan Right",
        "action_motion": "Character flying rapidly above crashing ocean waves, water spray flying, dynamic side tracking shot",
        "suggested_fx": "None",
        "suggested_sfx": "🌊 Ocean / River Waves",
        "visual": "Hanuman soaring across roaring blue ocean under golden sunset sky, crashing waves and mist below",
    },
    {
        "narration": "Lifting the entire Dronagiri mountain glowing with life-saving herbs, he illuminates the night sky.",
        "kannada_narration": "ಸಂಜೀವಿನಿ ಪರ್ವತವನ್ನೇ ಎತ್ತಿ ಹಿಡಿದು ಹನುಮಂತನು ರಾತ್ರಿಯ ಆಕಾಶವನ್ನು ಬೆಳಗಿಸುತ್ತಾನೆ.",
        "emotion": "Triumphant & Divine",
        "motion": "Zoom Out",
        "action_motion": "Hero holding a massive glowing mountain aloft, swirling green and gold magical particles rising upward",
        "suggested_fx": "Cinematic Vignette",
        "suggested_sfx": "⚡ Thunder Strike",
        "visual": "Hanuman carrying the glowing Sanjeevani mountain in one hand through starry night clouds, radiant light",
    },
    {
        "narration": "In the serene Himalayan forest, he meditates in eternal devotion as cosmic light surrounds him.",
        "kannada_narration": "ಶಾಂತ ಹಿಮಾಲಯದ ಕಾಡಿನಲ್ಲಿ ದಿವ್ಯ ಬೆಳಕಿನ ನಡುವೆ ಹನುಮಂತನು ಧ್ಯಾನಮಗ್ನನಾಗುತ್ತಾನೆ.",
        "emotion": "Peaceful & Sacred",
        "motion": "Zoom In",
        "action_motion": "Subtle breathing motion, gentle wind blowing leaves and hair, floating golden light motes around meditating hero",
        "suggested_fx": "None",
        "suggested_sfx": "🌲 Forest Birds",
        "visual": "Hanuman meditating peacefully on a Himalayan rock beside a sacred river and snow peaks, soft morning sunrays",
    },
]

# --- SESSION STATE INIT ---
for key, default in [
    ("scenes", [dict(s) for s in DEFAULT_SCENES]),
    ("images", []),
    ("scene_videos", []),
    ("uploaded_photo_paths", []),
    ("uploaded_video_clips", []),
    ("audio_files", []),
    ("bgm_audio_path", ""),
    ("bgm_volume", 0.20),
    ("bgm_duck_enabled", True),
    ("bgm_duck_volume", 0.07),
    ("voice_volume", 1.0),
    ("sfx_volume", 0.75),
    ("locked_character_prompt", DEFAULT_CHAR_PROMPT),
    ("lock_character_enabled", True),
    ("animation_style_name", "3D Pixar / Disney Feature Film"),
    ("gen_batch_id", 0),
    ("selected_clip_idx", 0),
    ("format_preset", "9:16 Reels / Shorts"),
    ("bg_fit_mode", "Crop Fill"),
    ("playback_speed", 1.0),
    ("custom_title_text", ""),
    ("subtitle_lang", "English"),
    ("subtitle_position", "Bottom"),
    ("subtitle_scale", 0.85),
    ("editor_brightness", 1.0),
    ("editor_contrast", 1.05),
    ("editor_saturation", 1.12),
    ("editor_blur", 0.0),
    ("editor_transition", "Crossfade"),
    ("editor_transition_sec", 0.4),
    ("editor_video_fx", "None"),
    ("scene_audio_fx", {}),
    ("scene_fx_overrides", {}),
    ("undo_stack", []),
    ("redo_stack", []),
    ("last_history_label", ""),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# --- HELPER FUNCTIONS ---
def _scene_widget_key(prefix, idx):
    return f"{prefix}_{idx}_b{st.session_state.gen_batch_id}"


def force_sync_scene_widget_states(scenes_list):
    for idx, sc in enumerate(scenes_list):
        st.session_state[_scene_widget_key("nar", idx)] = sc.get("narration", "")
        st.session_state[_scene_widget_key("kn_script", idx)] = sc.get("kannada_narration", "")
        st.session_state[_scene_widget_key("vis", idx)] = sc.get("visual", "")
        st.session_state[_scene_widget_key("act", idx)] = sc.get("action_motion", "")
        st.session_state[_scene_widget_key("emo", idx)] = sc.get("emotion", "Expressive")
        st.session_state[_scene_widget_key("mot", idx)] = sc.get("motion", "Auto Alternate")
        st.session_state[_scene_widget_key("fx_sc", idx)] = st.session_state.scene_fx_overrides.get(
            idx, sc.get("suggested_fx", "None")
        )


def record_undo_step(action_label="Edit"):
    snap = {
        "label": action_label,
        "scenes": copy.deepcopy(st.session_state.scenes),
        "images": list(st.session_state.images),
        "scene_videos": list(st.session_state.scene_videos),
        "uploaded_photo_paths": list(st.session_state.uploaded_photo_paths),
        "audio_files": list(st.session_state.audio_files),
    }
    u_stack = list(st.session_state.undo_stack)
    u_stack.append(snap)
    st.session_state.undo_stack = u_stack[-MAX_HISTORY_STEPS:]
    st.session_state.redo_stack = []
    st.session_state.last_history_label = action_label


def perform_undo():
    if not st.session_state.undo_stack:
        return False
    snap = st.session_state.undo_stack.pop()
    st.session_state.redo_stack.append({
        "label": "Redo",
        "scenes": copy.deepcopy(st.session_state.scenes),
        "images": list(st.session_state.images),
        "scene_videos": list(st.session_state.scene_videos),
        "uploaded_photo_paths": list(st.session_state.uploaded_photo_paths),
        "audio_files": list(st.session_state.audio_files),
    })
    for k in ("scenes", "images", "scene_videos", "uploaded_photo_paths", "audio_files"):
        st.session_state[k] = copy.deepcopy(snap[k])
    force_sync_scene_widget_states(st.session_state.scenes)
    st.session_state.last_history_label = f"Undid: {snap['label']}"
    return True


def perform_redo():
    if not st.session_state.redo_stack:
        return False
    snap = st.session_state.redo_stack.pop()
    for k in ("scenes", "images", "scene_videos", "uploaded_photo_paths", "audio_files"):
        st.session_state[k] = copy.deepcopy(snap[k])
    force_sync_scene_widget_states(st.session_state.scenes)
    st.session_state.last_history_label = "Redid change"
    return True


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


def apply_duration(clip, duration):
    return clip.with_duration(duration) if MOVIEPY_V2 else clip.set_duration(duration)


def apply_start(clip, start_t):
    return clip.with_start(start_t) if MOVIEPY_V2 else clip.set_start(start_t)


def apply_audio(clip, audio):
    return clip.with_audio(audio) if MOVIEPY_V2 else clip.set_audio(audio)


def subclip_safe(clip, start_t, end_t):
    s_val = max(0.0, float(start_t))
    e_val = max(s_val + 0.1, float(end_t))
    if clip.duration is not None:
        e_val = min(float(clip.duration), e_val)
        s_val = min(s_val, max(0.0, e_val - 0.1))
    if MOVIEPY_V2 and hasattr(clip, "subclipped"):
        return clip.subclipped(s_val, e_val)
    return clip.subclip(s_val, e_val)


def scale_audio_volume(clip, factor):
    if clip is None or abs(factor - 1.0) < 1e-3:
        return clip
    if MOVIEPY_V2 and hasattr(clip, "with_volume_scaled"):
        return clip.with_volume_scaled(factor)
    if hasattr(clip, "volumex"):
        return clip.volumex(factor)
    return clip


def ensure_exact_size(pil_img, target_w, target_h, bg_mode="Crop Fill"):
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    rgb = pil_img.convert("RGB")
    if rgb.size == (target_w, target_h):
        return rgb
    if bg_mode == "Blur Canvas":
        bg_scale = max(target_w / max(1, rgb.width), target_h / max(1, rgb.height))
        bg = rgb.resize((int(rgb.width * bg_scale), int(rgb.height * bg_scale)), resampling)
        bx = max(0, (bg.width - target_w) // 2)
        by = max(0, (bg.height - target_h) // 2)
        bg = bg.crop((bx, by, bx + target_w, by + target_h)).filter(ImageFilter.GaussianBlur(radius=16))
        bg = ImageEnhance.Brightness(bg).enhance(0.55)
        fg_scale = min(target_w / max(1, rgb.width), target_h / max(1, rgb.height))
        fg = rgb.resize((max(1, int(rgb.width * fg_scale)), max(1, int(rgb.height * fg_scale))), resampling)
        bg.paste(fg, ((target_w - fg.width) // 2, (target_h - fg.height) // 2))
        return bg

    scale = max(target_w / max(1, rgb.width), target_h / max(1, rgb.height))
    nw = max(target_w, int(round(rgb.width * scale)))
    nh = max(target_h, int(round(rgb.height * scale)))
    resized = rgb.resize((nw, nh), resampling)
    left = max(0, (nw - target_w) // 2)
    top = max(0, (nh - target_h) // 2)
    return resized.crop((left, top, left + target_w, top + target_h)).resize((target_w, target_h), resampling)


def apply_pil_effects(pil_img, brightness=1.0, contrast=1.0, saturation=1.0, blur_radius=0.0, video_fx="None"):
    out = pil_img.convert("RGB")
    if abs(brightness - 1.0) > 1e-3:
        out = ImageEnhance.Brightness(out).enhance(brightness)
    if abs(contrast - 1.0) > 1e-3:
        out = ImageEnhance.Contrast(out).enhance(contrast)
    if abs(saturation - 1.0) > 1e-3:
        out = ImageEnhance.Color(out).enhance(saturation)
    if blur_radius and blur_radius > 0.05:
        out = out.filter(ImageFilter.GaussianBlur(radius=float(blur_radius)))
    if video_fx == "Vintage Sepia":
        sepia = (0.393, 0.769, 0.189, 0, 0.349, 0.686, 0.168, 0, 0.272, 0.534, 0.131, 0)
        out = out.convert("RGB", sepia)
    elif video_fx == "Cinematic Vignette":
        w, h = out.size
        vig = Image.new("L", (w, h), 255)
        draw = ImageDraw.Draw(vig)
        draw.ellipse([-w * 0.2, -h * 0.2, w * 1.2, h * 1.2], fill=0)
        vig = vig.filter(ImageFilter.GaussianBlur(radius=max(w, h) * 0.25))
        out = Image.composite(out, Image.new("RGB", (w, h), (0, 0, 0)), vig)
    elif video_fx == "Flash White":
        out = ImageEnhance.Brightness(out).enhance(1.28)
    elif video_fx == "Glitch RGB":
        r, g, b = out.split()
        r = ImageEnhance.Brightness(r).enhance(1.18)
        b = ImageEnhance.Contrast(b).enhance(1.22)
        out = Image.merge("RGB", (r, g, b))
    return out


# --- FIX 1: YOUTUBE & DIRECT MEDIA DOWNLOADER ---
def download_media_or_youtube(raw_url, media_kind="video", prefix="media"):
    cleaned = (raw_url or "").strip()
    if not cleaned:
        raise ValueError("Empty URL provided.")
    os.makedirs("uploaded_studio_media", exist_ok=True)
    is_yt = any(d in cleaned.lower() for d in ("youtube.com", "youtu.be"))

    if is_yt:
        try:
            import yt_dlp
        except ImportError as exc:
            raise RuntimeError("Add 'yt-dlp' to requirements.txt to download YouTube links.") from exc

        out_tmpl = os.path.join("uploaded_studio_media", f"{prefix}_yt_{int(time.time())}.%(ext)s")
        ydl_opts = {
            "outtmpl": out_tmpl,
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
            "format": "best[ext=mp4][height<=720]/best[ext=mp4]/best"
            if media_kind == "video"
            else "bestaudio[ext=m4a]/bestaudio/best",
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(cleaned, download=True)
            downloaded_path = ydl.prepare_filename(info)
            if not os.path.exists(downloaded_path):
                base_p = os.path.splitext(downloaded_path)[0]
                for ext in (".mp4", ".webm", ".mkv", ".m4a", ".mp3"):
                    if os.path.exists(base_p + ext):
                        downloaded_path = base_p + ext
                        break
            if os.path.exists(downloaded_path) and os.path.getsize(downloaded_path) > 4096:
                return downloaded_path
            raise RuntimeError("YouTube download produced an empty file (video may be age/region restricted).")

    # Direct HTTP(S) media download
    resp = requests.get(cleaned, headers={"User-Agent": "Mozilla/5.0"}, stream=True, timeout=60)
    if not resp.ok:
        raise RuntimeError(f"HTTP {resp.status_code} fetching {cleaned}")
    ct = (resp.headers.get("Content-Type") or "").lower()
    if "text/html" in ct:
        raise RuntimeError("URL points to a webpage (HTML), not a direct MP4/MP3 file.")
    ext = ".mp4" if media_kind == "video" else ".mp3"
    out_path = os.path.join("uploaded_studio_media", f"{prefix}_{int(time.time() * 1000)}{ext}")
    with open(out_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=131072):
            if chunk:
                f.write(chunk)
    if os.path.getsize(out_path) < 2048:
        raise RuntimeError("Downloaded file is empty or invalid.")
    return out_path


# --- FIX 2: HIGH-DETAIL AI KEYFRAME GENERATOR (VALID MODEL IDS) ---
def generate_ai_keyframe(
    engine_choice,
    cf_account_id,
    cf_api_token,
    hf_token,
    gemini_key,
    prompt_text,
    ratio_str,
    target_w,
    target_h,
    ref_photo_path=None,
    seed=42,
):
    errors = []
    clean_p = " ".join((prompt_text or "").split())[:1500]

    def run_hf():
        if not hf_token or not hf_token.strip():
            raise RuntimeError("Hugging Face token missing")
        client = InferenceClient(token=hf_token.strip())
        w_hf = max(512, min(1024, (target_w // 8) * 8))
        h_hf = max(512, min(1024, (target_h // 8) * 8))
        for model_id in ["black-forest-labs/FLUX.1-schnell", "stabilityai/stable-diffusion-xl-base-1.0"]:
            try:
                img = client.text_to_image(prompt=clean_p, model=model_id, width=w_hf, height=h_hf)
                return ensure_exact_size(img, target_w, target_h), f"HF {model_id}"
            except Exception as e:
                last_e = e
        raise RuntimeError(str(last_e))

    def run_cf():
        if not (cf_account_id and cf_account_id.strip() and cf_api_token and cf_api_token.strip()):
            raise RuntimeError("Cloudflare credentials missing")
        url = f"https://api.cloudflare.com/client/v4/accounts/{cf_account_id.strip()}/ai/run/@cf/black-forest-labs/flux-1-schnell"
        headers = {"Authorization": f"Bearer {cf_api_token.strip()}", "Content-Type": "application/json"}
        resp = requests.post(url, headers=headers, json={"prompt": clean_p}, timeout=90)
        if not resp.ok:
            raise RuntimeError(f"CF HTTP {resp.status_code}: {resp.text[:200]}")
        if "image/" in resp.headers.get("Content-Type", ""):
            return ensure_exact_size(Image.open(BytesIO(resp.content)), target_w, target_h), "Cloudflare FLUX.1"
        b64 = resp.json().get("result", {}).get("image")
        if b64:
            return ensure_exact_size(Image.open(BytesIO(base64.b64decode(b64))), target_w, target_h), "Cloudflare FLUX.1"
        raise RuntimeError("Cloudflare returned no image")

    def run_gemini():
        if not gemini_key or not gemini_key.strip():
            raise RuntimeError("Gemini API key missing")
        client = genai.Client(api_key=gemini_key.strip())
        # Valid Gemini Image Generation models
        for model_id in ["gemini-2.5-flash-image", "gemini-2.0-flash-exp-image-generation"]:
            try:
                contents = [clean_p]
                if ref_photo_path and os.path.exists(ref_photo_path):
                    with Image.open(ref_photo_path) as ref_im:
                        contents = [
                            ref_im.convert("RGB"),
                            f"Keep the exact same main character appearance from this reference image. {clean_p}",
                        ]
                resp = client.models.generate_content(
                    model=model_id,
                    contents=contents,
                    config=types.GenerateContentConfig(response_modalities=["IMAGE"]),
                )
                for cand in getattr(resp, "candidates", None) or []:
                    for part in getattr(getattr(cand, "content", None), "parts", None) or []:
                        inline = getattr(part, "inline_data", None)
                        if inline and getattr(inline, "data", None):
                            return ensure_exact_size(Image.open(BytesIO(inline.data)), target_w, target_h), f"Gemini {model_id}"
            except Exception as e:
                last_e = e
        raise RuntimeError(str(last_e))

    def run_pollinations():
        w_p = max(384, min(1024, (target_w // 8) * 8))
        h_p = max(384, min(1024, (target_h // 8) * 8))
        enc = urllib.parse.quote(clean_p[:350], safe="")
        url = f"https://image.pollinations.ai/prompt/{enc}?width={w_p}&height={h_p}&seed={seed}&model=flux&nologo=true"
        r = requests.get(url, timeout=50)
        if r.ok and "image/" in r.headers.get("Content-Type", ""):
            return ensure_exact_size(Image.open(BytesIO(r.content)), target_w, target_h), "Pollinations FLUX"
        raise RuntimeError(f"Pollinations HTTP {r.status_code}")

    order = [
        ("Cloudflare", run_cf),
        ("Hugging Face", run_hf),
        ("Gemini", run_gemini),
        ("Pollinations", run_pollinations),
    ]
    if "Hugging Face" in engine_choice:
        order = [("Hugging Face", run_hf), ("Cloudflare", run_cf), ("Gemini", run_gemini), ("Pollinations", run_pollinations)]
    elif "Gemini" in engine_choice:
        order = [("Gemini", run_gemini), ("Cloudflare", run_cf), ("Hugging Face", run_hf), ("Pollinations", run_pollinations)]

    for name, fn in order:
        try:
            img, src = fn()
            return img, src, errors
        except Exception as exc:
            errors.append(f"{name}: {exc}")

    # If user uploaded reference photos, use them instead of drawing fake shapes
    if ref_photo_path and os.path.exists(ref_photo_path):
        with Image.open(ref_photo_path) as im:
            return ensure_exact_size(im, target_w, target_h), "Uploaded Reference Photo", errors

    return None, None, errors


# --- FIX 3: TRUE AI VIDEO GENERATION (TEXT-TO-VIDEO & IMAGE-TO-VIDEO) ---
def generate_true_ai_video_clip(
    video_engine,
    hf_token,
    gemini_key,
    fal_key,
    prompt_text,
    action_motion,
    keyframe_path,
    out_mp4_path,
    target_w,
    target_h,
):
    """Generates real AI-animated MP4 clips via cloud diffusion video models."""
    errors = []
    full_motion_prompt = f"{prompt_text}. Action & camera movement: {action_motion}. Smooth 24fps animation."[:1000]

    # 1. Hugging Face Serverless Video Models (Wan 2.1 / LTX-Video / CogVideoX)
    if hf_token and hf_token.strip() and ("Hugging Face" in video_engine or "Auto" in video_engine):
        client = InferenceClient(token=hf_token.strip())
        # Try Image-to-Video first if we have a keyframe image
        if keyframe_path and os.path.exists(keyframe_path) and hasattr(client, "image_to_video"):
            with open(keyframe_path, "rb") as f:
                img_bytes = f.read()
            for i2v_model in ["Wan-AI/Wan2.1-I2V-14B-480P", "Lightricks/LTX-Video"]:
                try:
                    vid_bytes = client.image_to_video(
                        image=img_bytes,
                        prompt=full_motion_prompt,
                        model=i2v_model,
                    )
                    if vid_bytes and len(vid_bytes) > 5000:
                        with open(out_mp4_path, "wb") as out_f:
                            out_f.write(vid_bytes)
                        return out_mp4_path, f"HF I2V ({i2v_model})", errors
                except Exception as exc:
                    errors.append(f"HF I2V {i2v_model}: {exc}")

        # Fallback to Text-to-Video on Hugging Face
        if hasattr(client, "text_to_video"):
            for t2v_model in ["Wan-AI/Wan2.1-T2V-1.3B", "THUDM/CogVideoX-5b", "Lightricks/LTX-Video"]:
                try:
                    vid_bytes = client.text_to_video(prompt=full_motion_prompt, model=t2v_model)
                    if vid_bytes and len(vid_bytes) > 5000:
                        with open(out_mp4_path, "wb") as out_f:
                            out_f.write(vid_bytes)
                        return out_mp4_path, f"HF T2V ({t2v_model})", errors
                except Exception as exc:
                    errors.append(f"HF T2V {t2v_model}: {exc}")

    # 2. Gemini Veo Video Generation (veo-2.0-generate-001)
    if gemini_key and gemini_key.strip() and ("Gemini Veo" in video_engine or "Auto" in video_engine):
        try:
            client = genai.Client(api_key=gemini_key.strip())
            aspect = "9:16" if target_h > target_w else "16:9"
            op = client.models.generate_videos(
                model="veo-2.0-generate-001",
                prompt=full_motion_prompt,
                config=types.GenerateVideosConfig(aspect_ratio=aspect, person_generation="allow_adult"),
            )
            waited = 0
            while not op.done and waited < 180:
                time.sleep(6)
                waited += 6
                op = client.operations.get(op)
            if op.done and getattr(op.response, "generated_videos", None):
                gen_vid = op.response.generated_videos[0]
                client.files.download(file=gen_vid.video)
                gen_vid.video.save(out_mp4_path)
                if os.path.exists(out_mp4_path) and os.path.getsize(out_mp4_path) > 5000:
                    return out_mp4_path, "Google Veo 2.0", errors
        except Exception as exc:
            errors.append(f"Gemini Veo: {exc}")

    # 3. fal.ai Fast Image-to-Video (Kling / Wan / Luma)
    if fal_key and fal_key.strip() and ("fal.ai" in video_engine or "Auto" in video_engine):
        try:
            headers = {"Authorization": f"Key {fal_key.strip()}", "Content-Type": "application/json"}
            payload = {"prompt": full_motion_prompt}
            if keyframe_path and os.path.exists(keyframe_path):
                with open(keyframe_path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("utf-8")
                payload["image_url"] = f"data:image/png;base64,{b64}"
                endpoint = "https://fal.run/fal-ai/wan-i2v"
            else:
                endpoint = "https://fal.run/fal-ai/wan-t2v"
            r = requests.post(endpoint, headers=headers, json=payload, timeout=180)
            if r.ok:
                vid_url = r.json().get("video", {}).get("url")
                if vid_url:
                    vr = requests.get(vid_url, timeout=90)
                    with open(out_mp4_path, "wb") as f:
                        f.write(vr.content)
                    return out_mp4_path, "fal.ai Wan Video", errors
            else:
                errors.append(f"fal.ai HTTP {r.status_code}: {r.text[:180]}")
        except Exception as exc:
            errors.append(f"fal.ai: {exc}")

    return None, None, errors


# --- FIX 4: 2.5D DEPTH PARALLAX + PARTICLE ANIMATION FALLBACK (NO STICK FIGURES!) ---
def build_25d_parallax_clip(
    img_path,
    out_path,
    target_w,
    target_h,
    duration=4.0,
    motion_mode="Zoom In",
    scene_idx=0,
    brightness=1.0,
    contrast=1.05,
    saturation=1.12,
    blur_radius=0.0,
    video_fx="None",
    bg_mode="Crop Fill",
):
    """Creates fluid 2.5D foreground/background separation + golden light particles + camera motion."""
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    safe_dur = max(1.0, float(duration))
    modes = ["Zoom In", "Pan Right", "Zoom Out", "Pan Left"]
    active_mode = modes[scene_idx % len(modes)] if motion_mode == "Auto Alternate" else motion_mode

    with Image.open(img_path) as im:
        base_rgb = ensure_exact_size(im.convert("RGB"), target_w, target_h, bg_mode=bg_mode)
    base_rgb = apply_pil_effects(base_rgb, brightness, contrast, saturation, blur_radius, video_fx)

    # Create depth layers: soft atmospheric background + sharp subject center
    overscan = 1.18
    cw, ch = int(target_w * overscan), int(target_h * overscan)
    fg_canvas = base_rgb.resize((cw, ch), resampling)
    bg_canvas = fg_canvas.filter(ImageFilter.GaussianBlur(radius=6))

    # Radial mask blending sharp foreground with soft depth background
    mask = Image.new("L", (cw, ch), 0)
    mdraw = ImageDraw.Draw(mask)
    mdraw.ellipse([int(cw * 0.12), int(ch * 0.10), int(cw * 0.88), int(ch * 0.90)], fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(radius=int(min(cw, ch) * 0.12)))
    layered_canvas = Image.composite(fg_canvas, bg_canvas, mask)

    # Pre-seed floating cinematic light particles (embers / dust motes)
    rng = random.Random(100 + scene_idx)
    particles = [
        (
            rng.uniform(0.05, 0.95),
            rng.uniform(0.10, 0.95),
            rng.uniform(0.04, 0.14),
            rng.randint(2, 5),
            rng.randint(130, 220),
        )
        for _ in range(22)
    ]

    def make_frame(t):
        u = max(0.0, min(1.0, float(t) / safe_dur))
        ease = u * u * (3.0 - 2.0 * u)

        if video_fx == "Zoom Punch":
            scale = 1.0 + 0.12 * math.sin(u * math.pi)
        elif active_mode == "Zoom In":
            scale = 1.0 + 0.14 * ease
        elif active_mode == "Zoom Out":
            scale = 1.15 - 0.14 * ease
        else:
            scale = 1.10

        win_w = max(2, int(round(target_w / scale)))
        win_h = max(2, int(round(target_h / scale)))

        if active_mode == "Pan Right":
            x0 = int(round((cw - win_w) * ease))
            y0 = (ch - win_h) // 2 + int(round(math.sin(u * math.pi * 2) * 8))
        elif active_mode == "Pan Left":
            x0 = int(round((cw - win_w) * (1.0 - ease)))
            y0 = (ch - win_h) // 2 + int(round(math.cos(u * math.pi * 2) * 8))
        else:
            x0 = (cw - win_w) // 2 + int(round(math.sin(u * math.pi * 2) * 12))
            y0 = (ch - win_h) // 2 + int(round(math.cos(u * math.pi * 2) * 8))

        if video_fx == "Shake":
            x0 += int(round(math.sin(t * 32.0) * 8))
            y0 += int(round(math.cos(t * 36.0) * 6))

        x0 = max(0, min(cw - win_w, x0))
        y0 = max(0, min(ch - win_h, y0))
        frame = layered_canvas.crop((x0, y0, x0 + win_w, y0 + win_h)).resize((target_w, target_h), resampling)

        # Render animated golden embers / volumetric light particles
        ov = Image.new("RGBA", (target_w, target_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(ov)
        for px0, py0, speed, rad, alpha in particles:
            px = int(((px0 + math.sin(u * 4.0 + py0 * 6.0) * 0.03) % 1.0) * target_w)
            py = int(((py0 - u * speed) % 1.0) * target_h)
            draw.ellipse([px - rad, py - rad, px + rad, py + rad], fill=(255, 215, 130, alpha))

        composed = Image.alpha_composite(frame.convert("RGBA"), ov).convert("RGB")
        return np.asarray(composed, dtype=np.uint8)

    from moviepy.video.VideoClip import VideoClip as DynamicVideoClip

    clip = apply_duration(DynamicVideoClip(make_frame, duration=safe_dur), safe_dur)
    try:
        clip.write_videofile(
            out_path,
            fps=ANIMATION_FPS,
            codec="libx264",
            audio=False,
            ffmpeg_params=["-crf", "19", "-preset", "veryfast", "-pix_fmt", "yuv420p"],
            logger=None,
        )
        return out_path
    finally:
        try:
            clip.close()
        except Exception:
            pass


# --- SUBTITLES & FONTS ---
@st.cache_resource(show_spinner=False)
def _cached_font(font_size: int, lang: str = "English"):
    if lang == "Kannada":
        tmp_font = "/tmp/NotoSansKannada-Regular.ttf"
        if not os.path.exists(tmp_font):
            try:
                r = requests.get(
                    "https://github.com/google/fonts/raw/main/ofl/notosanskannada/static/NotoSansKannada-Regular.ttf",
                    timeout=20,
                )
                if r.ok and len(r.content) > 40000:
                    with open(tmp_font, "wb") as f:
                        f.write(r.content)
            except Exception:
                pass
        if os.path.exists(tmp_font):
            return ImageFont.truetype(tmp_font, int(font_size))
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "arial.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, int(font_size))
    return ImageFont.load_default()


def draw_subtitle_on_frame(frame_pil, subtitle_text, target_w, target_h, lang="English", position="Bottom", scale=0.85):
    if not subtitle_text or not subtitle_text.strip() or lang == "None":
        return frame_pil
    img_rgba = frame_pil.convert("RGBA")
    draw = ImageDraw.Draw(img_rgba)
    font_size = max(18, int(round(min(target_w, target_h) * 0.030 * float(scale))))
    font = _cached_font(font_size, lang=lang)
    chars_per_line = max(18, int((target_w * 0.84) / max(9, font_size * 0.56)))
    lines = textwrap.wrap(subtitle_text.strip(), width=chars_per_line)[:2]
    if not lines:
        return frame_pil
    line_h = font_size + 8
    box_h = len(lines) * line_h + 16
    pad_x = max(16, int(target_w * 0.05))
    if position == "Top":
        box_top = max(14, int(target_h * 0.04))
    elif position == "Center":
        box_top = max(14, (target_h - box_h) // 2)
    else:
        box_top = max(14, target_h - box_h - max(28, int(target_h * 0.05)))
    draw.rounded_rectangle([pad_x, box_top, target_w - pad_x, box_top + box_h], radius=12, fill=(8, 11, 18, 200))
    cur_y = box_top + 8
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        tx = max(pad_x + 10, (target_w - (bbox[2] - bbox[0])) // 2)
        draw.text((tx + 1, cur_y + 1), line, font=font, fill=(0, 0, 0, 240))
        draw.text((tx, cur_y), line, font=font, fill=(255, 252, 240, 255))
        cur_y += line_h
    return img_rgba.convert("RGB")


# --- AI SCRIPT DIRECTOR ---
def generate_ai_script(gemini_key, story_cmd, num_scenes, char_prompt, style_name):
    if not gemini_key or not gemini_key.strip():
        return None, "Add Gemini API key to generate custom AI scripts."
    client = genai.Client(api_key=gemini_key.strip())
    prompt = f"""
Create a {num_scenes}-scene animated movie/reel script.
Visual Style: {style_name}
Main Character Consistency Lock: {char_prompt}
User Story Command: {story_cmd}

Return ONLY a valid JSON array of {num_scenes} objects (no markdown fences) with keys:
- "narration": vivid English voiceover (1-2 sentences)
- "kannada_narration": natural Kannada translation (1-2 sentences)
- "emotion": mood label
- "motion": one of ["Zoom In", "Zoom Out", "Pan Right", "Pan Left"]
- "action_motion": specific physical character movement & camera motion for AI video generation (e.g., "Hero leaps through clouds swinging golden mace, dynamic tracking camera")
- "suggested_fx": one of ["None", "Zoom Punch", "Shake", "Flash White", "Cinematic Vignette"]
- "visual": detailed visual prompt describing exact character appearance, clothing, pose, lighting, and environment
"""
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


# --- UI HEADER ---
st.markdown(
    """
    <div class="studio-header">
        <div style="display:flex;align-items:center;gap:10px;">
            <span class="studio-pill">AI VIDEO & ANIMATION PRO</span>
            <span style="font-weight:700;font-size:17px;">True AI Text-to-Video, Image-to-Video & 2.5D Studio</span>
        </div>
        <div class="studio-sub">Wan 2.1 / Veo AI Motion • Reference Photo Character Lock • YouTube + Direct Media • Voice & BGM</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- SIDEBAR: API KEYS & CHARACTER LOCK ---
with st.sidebar:
    st.header("🎬 Format & AI Engines")
    st.session_state.format_preset = st.selectbox(
        "Video Format Preset",
        list(FORMAT_PRESETS.keys()),
        index=list(FORMAT_PRESETS.keys()).index(st.session_state.format_preset),
    )
    width, height, ratio_str = FORMAT_PRESETS[st.session_state.format_preset]
    st.caption(f"Resolution: **{width} × {height}** (`{ratio_str}`)")

    st.session_state.animation_style_name = st.selectbox(
        "Animation Art Style",
        list(ANIMATION_STYLES.keys()),
        index=list(ANIMATION_STYLES.keys()).index(st.session_state.animation_style_name),
    )

    with st.expander("🔑 AI Video & Image API Keys", expanded=True):
        gemini_key = st.text_input("Gemini API Key (Script + Imagen + Veo 2)", type="password")
        hf_token = st.text_input("Hugging Face Token (Wan 2.1 Video + FLUX)", type="password")
        cf_account_id = st.text_input("Cloudflare Account ID (Free Daily FLUX)", type="password")
        cf_api_token = st.text_input("Cloudflare API Token", type="password")
        fal_key = st.text_input("fal.ai API Key (Optional Fast Kling/Wan I2V)", type="password")

    st.divider()
    st.subheader("🎭 Character Consistency Lock")
    st.session_state.lock_character_enabled = st.checkbox(
        "Enforce Character Consistency Across Scenes", value=st.session_state.lock_character_enabled
    )
    st.session_state.locked_character_prompt = st.text_area(
        "Hero & Setting Description",
        value=st.session_state.locked_character_prompt,
        height=110,
    )

# --- MAIN TABS ---
tab1, tab2, tab3, tab4 = st.tabs([
    "1️⃣ Story & AI Director",
    "2️⃣ True AI Video & Media",
    "3️⃣ Voiceover & BGM",
    "4️⃣ Render Final Movie",
])

# === TAB 1: STORY & DIRECTOR ===
with tab1:
    st.markdown("### 1. Describe Your Story & Generate Scene-by-Scene Action Prompts")
    rough_story = st.text_area(
        "Story / Reel Concept:",
        value="Hanuman fighting in Lanka, crossing the ocean, lifting the glowing Sanjeevani mountain, and meditating in the Himalayas.",
        height=90,
    )
    num_scenes = st.slider("Number of Scenes", 2, 8, len(st.session_state.scenes))

    if st.button("✨ Generate AI Director Script & Action Prompts", type="primary", use_container_width=True):
        record_undo_step("Generate AI Script")
        with st.spinner("Directing scenes, character poses, and camera motion..."):
            scenes_out, err = generate_ai_script(
                gemini_key,
                rough_story,
                num_scenes,
                st.session_state.locked_character_prompt,
                st.session_state.animation_style_name,
            )
            if scenes_out:
                st.session_state.scenes = scenes_out
                st.session_state.gen_batch_id += 1
                force_sync_scene_widget_states(scenes_out)
                st.success(f"Generated {len(scenes_out)} animated scenes!")
                st.rerun()
            else:
                st.warning(f"Using built-in mythological template ({err})")

    for i, sc in enumerate(st.session_state.scenes):
        with st.expander(f"Scene {i + 1}: {sc.get('emotion', 'Scene')}", expanded=(i < 2)):
            c1, c2 = st.columns(2)
            with c1:
                sc["narration"] = st.text_area(
                    f"English Narration #{i + 1}",
                    value=sc.get("narration", ""),
                    key=_scene_widget_key("nar", i),
                )
                sc["kannada_narration"] = st.text_area(
                    f"Kannada Narration #{i + 1}",
                    value=sc.get("kannada_narration", ""),
                    key=_scene_widget_key("kn_script", i),
                )
            with c2:
                sc["visual"] = st.text_area(
                    f"Visual Keyframe Prompt #{i + 1}",
                    value=sc.get("visual", ""),
                    key=_scene_widget_key("vis", i),
                )
                sc["action_motion"] = st.text_input(
                    f"AI Video Action / Motion #{i + 1}",
                    value=sc.get("action_motion", "Dynamic character action and smooth cinematic camera motion"),
                    key=_scene_widget_key("act", i),
                )

# === TAB 2: TRUE AI VIDEO & MEDIA ===
with tab2:
    st.markdown("### 2. Generate Real AI Animated Clips (Keyframes + AI Video Motion)")

    m_col1, m_col2 = st.columns(2)
    with m_col1:
        up_photos = st.file_uploader(
            "Upload Character / Reference Photos (Used for Image-to-Video & Consistency)",
            type=["png", "jpg", "jpeg", "webp"],
            accept_multiple_files=True,
        )
        if st.button("➕ Save Reference Photos", use_container_width=True) and up_photos:
            os.makedirs("uploaded_studio_media", exist_ok=True)
            saved = []
            for idx_p, pf in enumerate(up_photos):
                ext = os.path.splitext(pf.name)[1].lower() or ".jpg"
                pp = os.path.join("uploaded_studio_media", f"ref_{int(time.time())}_{idx_p}{ext}")
                with open(pp, "wb") as f:
                    f.write(pf.getbuffer())
                saved.append(pp)
            st.session_state.uploaded_photo_paths.extend(saved)
            st.success(f"Saved {len(saved)} reference photo(s)!")

    with m_col2:
        yt_or_mp4_url = st.text_input(
            "Import Video Clip from YouTube URL or Direct MP4 Link:",
            placeholder="https://www.youtube.com/watch?v=... or https://.../clip.mp4",
        )
        if st.button("🌐 Download & Add Video Clip", use_container_width=True) and yt_or_mp4_url.strip():
            with st.spinner("Downloading video via yt-dlp / direct stream..."):
                try:
                    v_path = download_media_or_youtube(yt_or_mp4_url, media_kind="video", prefix="imported_clip")
                    st.session_state.scene_videos.append(v_path)
                    st.success(f"Imported video: `{os.path.basename(v_path)}`")
                except Exception as exc:
                    st.error(f"Download failed: {exc}")

    st.divider()
    g_col1, g_col2 = st.columns(2)
    with g_col1:
        img_engine = st.selectbox(
            "Keyframe Image Engine",
            ["Cloudflare Workers AI (Free Daily FLUX)", "Hugging Face (FLUX.1)", "Gemini Image"],
        )
    with g_col2:
        vid_engine = st.selectbox(
            "AI Video Motion Engine",
            [
                "Auto (Hugging Face Wan 2.1 / Gemini Veo / fal.ai -> 2.5D Fallback)",
                "Hugging Face Video (Wan 2.1 / CogVideoX)",
                "Gemini Veo 2.0",
                "fal.ai Image-to-Video",
                "2.5D Depth Parallax + Particles (Instant Free)",
            ],
        )

    if st.button("🚀 Generate Full AI Animated Clips for All Scenes", type="primary", use_container_width=True):
        record_undo_step("Generate AI Video Clips")
        os.makedirs("generated_keyframes", exist_ok=True)
        os.makedirs("generated_videos", exist_ok=True)

        style_desc = ANIMATION_STYLES[st.session_state.animation_style_name]
        char_lock = st.session_state.locked_character_prompt if st.session_state.lock_character_enabled else ""
        ref_pool = [p for p in st.session_state.uploaded_photo_paths if os.path.exists(p)]

        new_images = []
        new_videos = []
        progress = st.progress(0.0)
        status_box = st.empty()

        for idx, sc in enumerate(st.session_state.scenes):
            status_box.info(f"Scene {idx + 1}/{len(st.session_state.scenes)}: Generating AI keyframe & motion video...")
            full_prompt = f"{style_desc}. {char_lock}. {sc.get('visual', '')}"
            ref_img = ref_pool[idx % len(ref_pool)] if ref_pool else None

            # Step A: Generate high-res keyframe image
            kf_path = os.path.join("generated_keyframes", f"kf_{idx}_{int(time.time())}.png")
            pil_kf, img_src, img_errs = generate_ai_keyframe(
                img_engine,
                cf_account_id,
                cf_api_token,
                hf_token,
                gemini_key,
                full_prompt,
                ratio_str,
                width,
                height,
                ref_photo_path=ref_img,
                seed=500 + idx * 31,
            )
            if pil_kf is not None:
                pil_kf.save(kf_path)
                new_images.append(kf_path)
            else:
                kf_path = ""
                if img_errs:
                    st.warning(f"Scene {idx + 1} image fallback: {img_errs[0]}")

            # Step B: Generate true AI video clip (I2V / T2V) or 2.5D Parallax fallback
            vid_out = os.path.join("generated_videos", f"scene_{idx}_{int(time.time())}.mp4")
            generated_mp4 = None
            vid_src = ""

            if "2.5D Depth Parallax" not in vid_engine:
                generated_mp4, vid_src, vid_errs = generate_true_ai_video_clip(
                    vid_engine,
                    hf_token,
                    gemini_key,
                    fal_key,
                    full_prompt,
                    sc.get("action_motion", "Dynamic cinematic motion"),
                    kf_path,
                    vid_out,
                    width,
                    height,
                )
                if not generated_mp4 and vid_errs:
                    st.caption(f"Scene {idx + 1} AI video API note: {vid_errs[0]} → Using 2.5D Parallax Engine.")

            if generated_mp4 and os.path.exists(generated_mp4):
                new_videos.append(generated_mp4)
                st.toast(f"Scene {idx + 1}: Rendered via {vid_src}")
            elif kf_path and os.path.exists(kf_path):
                build_25d_parallax_clip(
                    kf_path,
                    vid_out,
                    width,
                    height,
                    duration=4.5,
                    motion_mode=sc.get("motion", "Zoom In"),
                    scene_idx=idx,
                    brightness=st.session_state.editor_brightness,
                    contrast=st.session_state.editor_contrast,
                    saturation=st.session_state.editor_saturation,
                    video_fx=sc.get("suggested_fx", "None"),
                    bg_mode=st.session_state.bg_fit_mode,
                )
                new_videos.append(vid_out)

            progress.progress((idx + 1) / len(st.session_state.scenes))

        st.session_state.images = new_images
        st.session_state.scene_videos = new_videos
        status_box.success(f"Generated {len(new_videos)} animated video clips!")

    # Preview generated clips
    if st.session_state.scene_videos:
        st.markdown("#### Generated Scene Clips Preview")
        cols = st.columns(min(4, len(st.session_state.scene_videos)))
        for idx, vp in enumerate(st.session_state.scene_videos):
            if os.path.exists(vp):
                with cols[idx % len(cols)]:
                    st.caption(f"Scene {idx + 1}")
                    st.video(vp)

# === TAB 3: VOICEOVER & BGM ===
with tab3:
    st.markdown("### 3. Neural AI Voiceover & Background Music")
    v_col1, v_col2 = st.columns(2)
    with v_col1:
        voice_choice = st.selectbox("Narrator Voice", list(TTS_VOICES.keys()))
    with v_col2:
        st.session_state.subtitle_lang = st.selectbox("Script & Subtitle Language", ["English", "Kannada", "None"])

    if st.button("🎙️ Generate Neural Voiceovers for All Scenes", type="primary", use_container_width=True):
        os.makedirs("generated_audio", exist_ok=True)
        voice_id = TTS_VOICES[voice_choice]
        aud_paths = []
        with st.spinner("Synthesizing neural narration..."):
            for idx, sc in enumerate(st.session_state.scenes):
                txt = (
                    sc.get("kannada_narration")
                    if st.session_state.subtitle_lang == "Kannada" and sc.get("kannada_narration")
                    else sc.get("narration", f"Scene {idx + 1}")
                )
                out_mp3 = os.path.join("generated_audio", f"voice_{idx}_{int(time.time())}.mp3")
                run_async(edge_tts.Communicate(txt, voice_id).save(out_mp3))
                aud_paths.append(out_mp3)
        st.session_state.audio_files = aud_paths
        st.success(f"Generated {len(aud_paths)} voiceover tracks!")

    bgm_file = st.file_uploader("Upload Background Music (MP3 / WAV)", type=["mp3", "wav", "m4a"])
    if bgm_file:
        os.makedirs("uploaded_studio_media", exist_ok=True)
        bp = os.path.join("uploaded_studio_media", f"bgm_{int(time.time())}.mp3")
        with open(bp, "wb") as f:
            f.write(bgm_file.getbuffer())
        st.session_state.bgm_audio_path = bp
        st.success("BGM loaded!")

# === TAB 4: FINAL MOVIE / REEL EXPORT ===
with tab4:
    st.markdown("### 4. Assemble & Export Final Animated Movie / Reel")
    if st.button("🎬 Render Final Animated MP4", type="primary", use_container_width=True):
        vids = [p for p in st.session_state.scene_videos if os.path.exists(p)]
        if not vids:
            st.error("Generate or import video clips in Tab 2 first.")
        else:
            os.makedirs("final_exports", exist_ok=True)
            final_mp4 = os.path.join("final_exports", f"AI_Animated_Story_{int(time.time())}.mp4")
            assembled_clips = []
            handles = []

            with st.spinner("Stitching AI clips, neural voiceover, subtitles, and BGM..."):
                for idx, vp in enumerate(vids):
                    vc = VideoFileClip(vp)
                    handles.append(vc)
                    aud_clip = None
                    if idx < len(st.session_state.audio_files) and os.path.exists(st.session_state.audio_files[idx]):
                        aud_clip = AudioFileClip(st.session_state.audio_files[idx])
                        handles.append(aud_clip)
                        target_dur = max(float(vc.duration or 3.0), float(aud_clip.duration or 3.0) + 0.3)
                    else:
                        target_dur = float(vc.duration or 4.0)

                    if (vc.duration or 0.0) < target_dur and hasattr(vc, "loop"):
                        vc = apply_duration(vc.loop(duration=target_dur), target_dur)
                    else:
                        vc = apply_duration(subclip_safe(vc, 0.0, target_dur), target_dur)

                    # Burn subtitles onto frames
                    sc_obj = st.session_state.scenes[idx] if idx < len(st.session_state.scenes) else {}
                    sub_txt = (
                        sc_obj.get("kannada_narration")
                        if st.session_state.subtitle_lang == "Kannada"
                        else sc_obj.get("narration", "")
                    )
                    if st.session_state.subtitle_lang != "None" and sub_txt:
                        def add_sub(gf, t, s_txt=sub_txt):
                            pil = ensure_exact_size(Image.fromarray(gf(t)), width, height, st.session_state.bg_fit_mode)
                            pil = draw_subtitle_on_frame(
                                pil, s_txt, width, height, st.session_state.subtitle_lang, st.session_state.subtitle_position
                            )
                            return np.asarray(pil, dtype=np.uint8)

                        if hasattr(vc, "transform"):
                            vc = vc.transform(add_sub)
                        elif hasattr(vc, "fl"):
                            vc = vc.fl(add_sub)

                    if aud_clip is not None:
                        vc = apply_audio(vc, aud_clip)
                    assembled_clips.append(vc)

                final_video = concatenate_videoclips(assembled_clips, method="compose")

                # Mix BGM if provided
                if st.session_state.bgm_audio_path and os.path.exists(st.session_state.bgm_audio_path):
                    bgm_raw = AudioFileClip(st.session_state.bgm_audio_path)
                    handles.append(bgm_raw)
                    tot_d = float(final_video.duration or 4.0)
                    bgm_loop = bgm_raw.loop(duration=tot_d) if hasattr(bgm_raw, "loop") else apply_duration(bgm_raw, tot_d)
                    bgm_scaled = scale_audio_volume(bgm_loop, 0.18)
                    if final_video.audio is not None:
                        final_video = apply_audio(final_video, CompositeAudioClip([final_video.audio, bgm_scaled]))
                    else:
                        final_video = apply_audio(final_video, bgm_scaled)

                final_video.write_videofile(
                    final_mp4,
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

            st.success("Final animated video rendered!")
            st.video(final_mp4)
            with open(final_mp4, "rb") as f:
                st.download_button(
                    "⬇️ Download Final Animated MP4",
                    data=f.read(),
                    file_name=os.path.basename(final_mp4),
                    mime="video/mp4",
                    use_container_width=True,
                )

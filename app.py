import asyncio
import json
import os
import tempfile
import urllib.parse
from io import BytesIO

import edge_tts
import fal_client
import requests
import streamlit as st
from google import genai
from moviepy.editor import AudioFileClip, ImageClip, VideoFileClip, concatenate_videoclips
from PIL import Image

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


st.title("🎬 AI Animated Story Studio")
st.caption("Turn rough story ideas into downloadable animated videos.")

with st.sidebar:
    st.header("🔑 Studio Settings")
    api_key = st.text_input("Gemini API Key", type="password")
    fal_key = st.text_input("fal.ai API Key (for moving clips)", type="password")
    i2v_model = st.selectbox(
        "Image-to-Video Model",
        [
            "fal-ai/wan-3.0-prime/image-to-video",
            "fal-ai/minimax/h3/image-to-video",
            "fal-ai/kling-video/v3/standard/image-to-video",
            "bytedance/seedance-2.0/image-to-video",
        ],
    )
    aspect_ratio = st.selectbox("Video Format", ["9:16 (Reels / Shorts)", "16:9 (YouTube)"])
    width, height = (720, 1280) if "9:16" in aspect_ratio else (1280, 720)

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
        placeholder="e.g., a small boy finds an injured glowing bird in forest. he takes care of it. later bird becomes giant phoenix and saves his village from flood.",
        height=140,
    )
    num_scenes = st.slider("Number of Scenes", 3, 6, 4)

    if st.button("✨ Polish Story & Create Scenes", type="primary"):
        if not api_key or not rough_story.strip():
            st.error("Please enter your Gemini API Key in the sidebar and write a rough story.")
        else:
            try:
                with st.spinner("Polishing grammar and structuring scenes..."):
                    client = genai.Client(api_key=api_key.strip())
                    prompt = f"""
                    Fix and rewrite this rough story into an engaging {num_scenes}-scene animated short script.
                    Return ONLY valid JSON as a list of objects, with no markdown formatting.
                    Each object must have:
                    - "narration": clear, emotional voiceover text (1-2 sentences)
                    - "visual": detailed visual action description for that scene (no character appearance details, just action and background)
                    Rough story: {rough_story.strip()}
                    """
                    response = client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=prompt,
                    )
                    raw_text = (response.text or "").strip()
                    if raw_text.startswith("```"):
                        lines = [line for line in raw_text.splitlines() if not line.strip().startswith("```")]
                        raw_text = "\n".join(lines).strip()

                    parsed_scenes = json.loads(raw_text)
                    if not isinstance(parsed_scenes, list) or len(parsed_scenes) == 0:
                        raise ValueError("Model did not return a valid scene list.")

                    st.session_state.scenes = [
                        {
                            "narration": str(item.get("narration", "")).strip(),
                            "visual": str(item.get("visual", "")).strip(),
                        }
                        for item in parsed_scenes
                    ]
                    st.success("Script ready! Review below or move to Tab 2.")
            except Exception as exc:
                st.error(f"Could not generate script: {exc}")

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
        placeholder="e.g., A 10-year-old Indian boy wearing a yellow kurta, big curious eyes, messy black hair",
    )

    if st.button("🎨 Generate Scene Artwork", type="primary"):
        if not st.session_state.scenes:
            st.warning("Please generate your script in Tab 1 first.")
        else:
            st.session_state.images = []
            st.session_state.scene_clips = []
            progress = st.progress(0)

            try:
                for idx, scene in enumerate(st.session_state.scenes):
                    full_prompt = f"{art_style}. Character: {char_desc}. Scene: {scene['visual']}"
                    encoded = urllib.parse.quote(full_prompt)
                    img_url = (
                        f"https://image.pollinations.ai/prompt/{encoded}"
                        f"?width={width}&height={height}&seed=42&nologo=true"
                    )
                    resp = requests.get(img_url, timeout=60)
                    resp.raise_for_status()

                    img_path = f"scene_{idx}.png"
                    with Image.open(BytesIO(resp.content)) as pil_img:
                        pil_img.convert("RGB").save(img_path)

                    st.session_state.images.append(img_path)
                    progress.progress((idx + 1) / len(st.session_state.scenes))

                st.success("All scene visuals generated!")
            except Exception as exc:
                st.error(f"Error generating scene images: {exc}")

    if st.session_state.images:
        cols = st.columns(2)
        for idx, img_path in enumerate(st.session_state.images):
            if os.path.exists(img_path):
                cols[idx % 2].image(img_path, caption=f"Scene {idx + 1}", use_container_width=True)

        st.divider()
        st.subheader("Animate Scene Images into Moving Clips")
        motion_hint = st.text_input(
            "Character & Camera Motion Hint",
            value="Expressive character movement, natural blinking and gesture, smooth cinematic camera motion",
        )

        if st.button("🎞️ Animate Scene Images into Clips"):
            if not fal_key.strip():
                st.warning("Enter your fal.ai API Key in the sidebar to run image-to-video generation.")
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
                    st.error(f"Image-to-video failed (check fal.ai credits and API key): {exc}")

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
                            scene_clip = vid.subclip(0, target_duration).set_audio(audio_clip)
                        else:
                            scene_clip = (
                                ImageClip(st.session_state.images[i])
                                .set_duration(target_duration)
                                .resize(lambda t: 1 + 0.04 * t)
                                .set_position(("center", "center"))
                                .set_audio(audio_clip)
                            )
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

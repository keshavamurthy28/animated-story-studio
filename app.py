import streamlit as st
from google import genai
import asyncio
import edge_tts
import requests
import json
import urllib.parse
from PIL import Image
from io import BytesIO
from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips

st.set_page_config(page_title="AI Animated Story Studio", page_icon="🎬", layout="centered")

# Initialize session memory so progress isn't lost between clicks
for key in ["scenes", "char_prompt", "images", "audio_files"]:
    if key not in st.session_state:
        st.session_state[key] = [] if key != "char_prompt" else ""

st.title("🎬 AI Animated Story Studio")
st.caption("Turn rough story ideas into downloadable animated videos.")

with st.sidebar:
    st.header("🔑 Studio Settings")
    api_key = st.text_input("Gemini API Key", type="password")
    aspect_ratio = st.selectbox("Video Format", ["9:16 (Reels / Shorts)", "16:9 (YouTube)"])
    width, height = (720, 1280) if "9:16" in aspect_ratio else (1280, 720)

tab1, tab2, tab3, tab4 = st.tabs([
    "1️⃣ Rough Story", "2️⃣ Characters & Art", "3️⃣ Voice & Sound", "4️⃣ Download Video"
])

# --- TAB 1: SCRIPT WRITER ---
with tab1:
    st.subheader("Write Your Rough Idea")
    rough_story = st.text_area(
        "Type your rough story in any style (even broken sentences):",
        placeholder="e.g., a small boy finds a injured glowing bird in forest. he takes care of it. later bird becomes giant phoenix and saves his village from flood.",
        height=140
    )
    num_scenes = st.slider("Number of Scenes", 3, 6, 4)

    if st.button("✨ Polish Story & Create Scenes", type="primary"):
        if not api_key or not rough_story:
            st.error("Please enter your Gemini API Key in the sidebar and write a rough story.")
        else:
            with st.spinner("Polishing grammar and structuring scenes..."):
                client = genai.Client(api_key=api_key)
                prompt = f"""
                Fix and rewrite this rough story into an engaging {num_scenes}-scene animated short script.
                Return ONLY valid JSON as a list of objects, with no markdown formatting.
                Each object must have:
                - "narration": clear, emotional voiceover text (1-2 sentences)
                - "visual": detailed visual action description for that scene (no character appearance details, just action and background)
                Rough story: {rough_story}
                """
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt
                )
                clean_json = response.text.strip().removeprefix("```json").removesuffix("```").strip()
                st.session_state.scenes = json.loads(clean_json)
                st.success("Script ready! Review below or move to Tab 2.")

    if st.session_state.scenes:
        for i, scene in enumerate(st.session_state.scenes):
            with st.expander(f"Scene {i+1}", expanded=True):
                scene["narration"] = st.text_area(f"Narration {i+1}", scene["narration"], key=f"nar_{i}")
                scene["visual"] = st.text_input(f"Visual Action {i+1}", scene["visual"], key=f"vis_{i}")

# --- TAB 2: CHARACTER SELECTION & ART ---
with tab2:
    st.subheader("Cast Your Characters & Art Style")
    art_style = st.selectbox("Choose Animation Style", [
        "3D Pixar Disney animation style, vibrant lighting, expressive faces",
        "2D Studio Ghibli anime style, soft watercolor backgrounds, warm",
        "Children's storybook illustration, textured paper, magical glow",
        "3D Unreal Engine cinematic fantasy animation"
    ])
    char_desc = st.text_input(
        "Describe your main character(s) (keeps them looking the same in every scene):",
        placeholder="e.g., A 10-year-old Indian boy wearing a yellow kurta, big curious eyes, messy black hair"
    )

    if st.button("🎨 Generate Scene Artwork", type="primary"):
        if not st.session_state.scenes:
            st.warning("Please generate your script in Tab 1 first.")
        else:
            st.session_state.images = []
            progress = st.progress(0)
            for idx, scene in enumerate(st.session_state.scenes):
                full_prompt = f"{art_style}. Character: {char_desc}. Scene: {scene['visual']}"
                encoded = urllib.parse.quote(full_prompt)
                img_url = f"https://image.pollinations.ai/prompt/{encoded}?width={width}&height={height}&seed=42&nologo=true"
                img_data = requests.get(img_url).content
                img_path = f"scene_{idx}.png"
                Image.open(BytesIO(img_data)).save(img_path)
                st.session_state.images.append(img_path)
                progress.progress((idx + 1) / len(st.session_state.scenes))
            st.success("All scene visuals generated!")

    if st.session_state.images:
        cols = st.columns(2)
        for idx, img_path in enumerate(st.session_state.images):
            cols[idx % 2].image(img_path, caption=f"Scene {idx+1}", use_container_width=True)

# --- TAB 3: VOICE & SOUND ---
with tab3:
    st.subheader("Select Narrator Voice")
    voice_map = {
        "Warm Male Storyteller (US)": "en-US-ChristopherNeural",
        "Expressive Female Storyteller (US)": "en-US-AriaNeural",
        "Indian Male Narrator": "en-IN-PrabhatNeural",
        "Indian Female Narrator": "en-IN-NeerjaNeural",
        "British Documentary Male": "en-GB-RyanNeural"
    }
    selected_voice = st.selectbox("Voice Options", list(voice_map.keys()))

    async def make_audio(text, voice, filename):
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(filename)

    if st.button("🎙️ Preview Sample Voice"):
        asyncio.run(make_audio("Once upon a time, a small spark changed everything.", voice_map[selected_voice], "sample.mp3"))
        st.audio("sample.mp3")

    if st.button("🔊 Generate Full Story Narration", type="primary"):
        if not st.session_state.scenes:
            st.warning("Generate your script in Tab 1 first.")
        else:
            st.session_state.audio_files = []
            for idx, scene in enumerate(st.session_state.scenes):
                aud_path = f"voice_{idx}.mp3"
                asyncio.run(make_audio(scene["narration"], voice_map[selected_voice], aud_path))
                st.session_state.audio_files.append(aud_path)
            st.success("All voiceovers recorded! Go to Tab 4 to render.")

# --- TAB 4: RENDER & DOWNLOAD ---
with tab4:
    st.subheader("Assemble & Download Final Movie")
    if st.button("🎬 Render Final MP4 Video", type="primary"):
        if not st.session_state.images or not st.session_state.audio_files:
            st.error("Make sure both Scene Artwork (Tab 2) and Narration (Tab 3) are generated.")
        else:
            with st.spinner("Stitching scenes, adding gentle motion, and exporting MP4..."):
                clips = []
                for img_p, aud_p in zip(st.session_state.images, st.session_state.audio_files):
                    audio_clip = AudioFileClip(aud_p)
                    # Smooth zoom-in motion effect so characters/scenes feel alive
                    img_clip = (ImageClip(img_p)
                                .set_duration(audio_clip.duration + 0.4)
                                .resize(lambda t: 1 + 0.04 * t)
                                .set_position(("center", "center"))
                                .set_audio(audio_clip))
                    clips.append(img_clip)

                final_video = concatenate_videoclips(clips, method="compose")
                output_file = "animated_story.mp4"
                final_video.write_videofile(output_file, fps=24, codec="libx264", audio_codec="aac")

            st.video(output_file)
            with open(output_file, "rb") as f:
                st.download_button(
                    label="⬇️ Download Video for Reels / YouTube",
                    data=f,
                    file_name="my_animated_story.mp4",
                    mime="video/mp4"
                )

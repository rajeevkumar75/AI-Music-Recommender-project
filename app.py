import os
import pickle
import streamlit as st
import spotipy
import faiss
import numpy as np
import pandas as pd
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv


load_dotenv()

st.set_page_config(
    page_title="AI Music Recommender 🎧",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<link rel="stylesheet"
      href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">
""", unsafe_allow_html=True)

st.markdown("""
<style>
.song-card {
    background: radial-gradient(circle at top left, #00ffc8 0, #111 45%);
    border-radius: 18px;
    padding: 12px;
    margin-bottom: 16px;
    text-align: center;
    box-shadow: 0 0 14px rgba(0, 255, 200, 0.25);
    border: 1px solid rgba(0, 255, 200, 0.25);
}
.song-cover {
    border-radius: 14px;
    width: 100%;
    aspect-ratio: 1 / 1;
    object-fit: cover;
}
.song-title { font-weight: 700; color: #fff; }
.song-artist { font-size: 13px; color: #b3b3b3; }
.song-album { font-size: 11px; color: #888; }
.song-spotify-btn {
    padding: 4px 12px;
    border-radius: 999px;
    background: black;
    color: black;
    text-decoration: none;
    font-size: 12px;
    font-weight: bold;
}
a { color: inherit; text-decoration: none; }
            
.song-spotify-btn:hover { background: #1db954; 
    text-decoration: none; 
    color: white;
    font-weight: bold; }

.seed-container {
    display: flex;
    gap: 12px;
    align-items: center;
    padding: 10px;
    border-radius: 12px;
    background-color: #1f1f1f;
    border: 1px solid #00ffe530;
    margin-top: 10px;
}
.seed-img {
    width: 60px;
    height: 60px;
    border-radius: 8px;
    object-fit: cover;
}
</style>
""", unsafe_allow_html=True)

#spotify credential:
CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")

if not CLIENT_ID or not CLIENT_SECRET:
    st.error("Spotify credentials not set")
    st.stop()

sp = spotipy.Spotify(
    client_credentials_manager=SpotifyClientCredentials(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET
    )
)

#load models
@st.cache_resource
def load_models():
    with open("models/df.pkl", "rb") as f:
        music_df = pickle.load(f)

    index = faiss.read_index("models/music_faiss.index")
    embeddings = np.load("models/music_embeddings.npy")

    return music_df.reset_index(drop=True), index, embeddings

music, faiss_index, embeddings = load_models()

#spotify details
@st.cache_data(show_spinner=False)
def get_song_details(song, artist):
    try:
        res = sp.search(q=f"track:{song} artist:{artist}", type="track", limit=1)
        item = res["tracks"]["items"][0]
        return {
            "image": item["album"]["images"][0]["url"],
            "preview": item["preview_url"],
            "link": item["external_urls"]["spotify"],
            "artist": item["artists"][0]["name"],
            "album": item["album"]["name"]
        }
    except:
        return None

#recommendation
def recommend(song, top_k, hide_no_preview):
    idx_list = music[music["song"] == song].index
    if len(idx_list) == 0:
        return []

    idx = idx_list[0]
    query = embeddings[idx].reshape(1, -1)
    faiss.normalize_L2(query)

    _, indices = faiss_index.search(query, top_k + 1)

    results = []
    for i in indices[0]:
        if i == idx:
            continue
        row = music.iloc[i]
        d = get_song_details(row.song, row.artist)
        if not d:
            continue
        if hide_no_preview and not d["preview"]:
            continue

        results.append((row.song, d))
        if len(results) == top_k:
            break

    return results

#for header
st.markdown("""
<h1 style='text-align:center; color:#00ffea;'>
<i class='fa-brands fa-spotify'></i> AI Music Recommender
</h1>
<p style='text-align:center; color:#cfcfcf;'>
FAISS-powered real-time music recommendations
</p>
""", unsafe_allow_html=True)

st.markdown("---")


tab_reco, tab_about = st.tabs(["🎧 Recommendations", "👨‍💻 About Me"])

#for recommendation tab
with tab_reco:

    with st.sidebar:
        st.markdown("### 🎵 Select Seed Song")

        artists = sorted(music["artist"].dropna().unique().tolist())
        selected_artist = st.selectbox(
            "Filter by Artist",
            ["All Artists"] + artists
        )

        if selected_artist == "All Artists":
            filtered_music = music
        else:
            filtered_music = music[music["artist"] == selected_artist]

        seed_song = st.selectbox(
            "Select Seed Song",
            filtered_music["song"].sort_values().unique().tolist()
        )

        if seed_song:
            seed_row = music[music["song"] == seed_song].iloc[0]
            seed_details = get_song_details(seed_song, seed_row.artist)
            if seed_details:
                st.markdown(f"""
                <div class="seed-container">
                    <img src="{seed_details['image']}" class="seed-img">
                    <div>
                        <b style="color:white">{seed_song}</b><br>
                        <span style="color:#b3b3b3">{seed_details['artist']}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("### ⚙ Advanced Filters")
        hide_no_preview = st.checkbox("Hide songs without preview", value=False)

        with st.expander("More options (Model dependent)"):
            st.slider("Year Range", 1950, 2025, (1990, 2025))
            st.select_slider("Mood", ["Any", "Chill", "Happy", "Energetic", "Sad"])

        num_recs = st.slider("Number of recommendations", 5, 25, 12)
        generate = st.button("Recommend 🎧")

    if generate and seed_song:
        with st.spinner("Finding similar tracks..."):
            results = recommend(seed_song, num_recs, hide_no_preview)

        cols = st.columns(4)
        for i, (name, d) in enumerate(results):
            with cols[i % 4]:
                st.markdown(f"""
                <div class="song-card">
                    <img src="{d['image']}" class="song-cover">
                    <p class="song-title">{name}</p>
                    <p class="song-artist">{d['artist']}</p>
                    <p class="song-album">{d['album']}</p>
                    <a href="{d['link']}" target="_blank" style="text-decoration: none; color: #00ffea; font-weight: bold;" class="song-spotify-btn">
                        Open in Spotify <i class="fa-brands fa-spotify"></i>
                    </a>
                </div>
                """, unsafe_allow_html=True)

                if d["preview"]:
                    st.audio(d["preview"])
    else:
        st.info("Select a song and click Generate 🎧")


#about tab
with tab_about:
    st.subheader("👋 About This Project")

    st.markdown("""
This is a **scalable AI music recommendation system** built using **embedding-based similarity search**.

- Replaced a **1.6 GB cosine similarity matrix** with **vector embeddings + FAISS** for fast and memory-efficient recommendations  
- Implemented **real-time music retrieval** using the **Spotify Web API** (album art, previews, metadata)  
- Designed a **low-latency, modular pipeline** suitable for **free cloud deployment**  
- Built an **interactive Streamlit interface** for real-time user interaction  
                


> Focused on **performance, scalability, and real-world ML system design**.
""")



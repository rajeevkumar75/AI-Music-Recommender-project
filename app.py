import os
import pickle
import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
load_dotenv()

#PAGE CONFIG:-
st.set_page_config(
    page_title="AI Music Recommender 🎧",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

#LOAD FONT AWESOME:--
st.markdown("""
<link rel="stylesheet"
      href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">
""", unsafe_allow_html=True)

#GLOBAL STYLES:-
st.markdown("""
<style>
.song-card {
    background: radial-gradient(circle at top left, #00ffc8 0, #111 45%);
    border-radius: 18px;
    padding: 12px 12px 16px 12px;
    margin-bottom: 16px;
    text-align: center;
    box-shadow: 0 0 14px rgba(0, 255, 200, 0.25);
    transition: transform 0.15s ease-out, box-shadow 0.15s ease-out, background 0.2s ease-out;
    border: 1px solid rgba(0, 255, 200, 0.25);
    height: 100%;
}
.song-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 0 20px rgba(0, 255, 200, 0.45);
    background: radial-gradient(circle at top left, #00ffe5 0, #121212 50%);
}
.song-cover {
    border-radius: 14px;
    width: 100%;
    aspect-ratio: 1 / 1;
    object-fit: cover;
    margin-bottom: 8px;
}
.song-title {
    font-weight: 700;
    font-size: 16px;
    color: #fdfdfd;
    margin: 0;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.song-artist {
    font-size: 13px;
    color: #b3b3b3;
    margin: 2px 0 2px 0;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.song-album {
    font-size: 11px;
    color: #888888;
    margin: 0 0 6px 0;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.song-tags {
    display: flex;
    justify-content: center;
    gap: 6px;
    margin-bottom: 6px;
    flex-wrap: wrap;
}

.song-tag {
    font-size: 11px;
    padding: 2px 8px;
    border-radius: 999px;
    background: rgba(0, 0, 0, 0.45);
    border: 1px solid rgba(0, 255, 200, 0.25);
    color: #cfcfcf;
}
.song-actions {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-top: 4px;
    font-size: 11px;
}
.song-preview-label {
    color: #b3b3b3;
}
.song-preview-label.ok {
    color: #1db954;
    font-weight: 600;
}
.song-spotify-btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 999px;
    background-color: black;
    color: #ffffff;
    font-size: 12px;
    font-weight: 600;
    text-decoration: none;
    box-shadow: 0 0 6px rgba(0,0,0,0.4);
}
.song-spotify-btn:hover {
    background-color: #1ed760;
    text-decoration: none;
}
.seed-song-container {
    display: flex;
    align-items: center;
    gap: 15px;
    margin-top: 15px;
    padding: 15px;
    border: 1px solid #00ffe520;
    border-radius: 12px;
    background-color: #1f1f1f;
}
.seed-song-image {
    width: 60px;
    height: 60px;
    border-radius: 8px;
    object-fit: cover;
}
.seed-song-info {
    flex-grow: 1;
}
.seed-song-title {
    font-size: 18px;
    font-weight: 700;
    color: #00ffea;
    margin: 0;
}
.seed-song-artist {
    font-size: 14px;
    color: #cfcfcf;
    margin: 0;
}
.block-container {
    padding-top: 1.5rem;
}
section.main .block-container {
    max-width: 1200px;
}
</style>
""", unsafe_allow_html=True)


#FOR STREAMIT SECRETE KEY:----
CLIENT_ID = st.secrets["SPOTIFY_CLIENT_ID"]
CLIENT_SECRET = st.secrets["SPOTIFY_CLIENT_SECRET"]

#SPOTIFY CREDENTIALS:-------
CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID", "YOUR_CLIENT_ID")
CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET", "YOUR_CLIENT_SECRET")

if CLIENT_ID in ("YOUR_CLIENT_ID", "", None) or CLIENT_SECRET in ("YOUR_CLIENT_SECRET", "", None):
    st.error("Spotify credentials not set. Please set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET environment variables.")
    st.stop()

try:
    sp = spotipy.Spotify(
        client_credentials_manager=SpotifyClientCredentials(
            client_id=CLIENT_ID,
            client_secret=CLIENT_SECRET
        )
    )
except Exception as e:
    st.error(f"Failed to initialize Spotify client: {e}")
    st.stop()

# ----------------- LOAD MODELS -----------------
@st.cache_data()
def load_models() -> tuple[pd.DataFrame, Any]:
    df_path = "models/df"
    sim_path = "models/music_recommender"

    if not os.path.exists(df_path) and os.path.exists(df_path + ".pkl"):
        df_path = df_path + ".pkl"
    if not os.path.exists(sim_path) and os.path.exists(sim_path + ".pkl"):
        sim_path = sim_path + ".pkl"

    if not os.path.exists(df_path) or not os.path.exists(sim_path):
        return pd.DataFrame({'song': [], 'artist': []}), None

    try:
        with open(df_path, "rb") as f:
            music_df = pickle.load(f)
        with open(sim_path, "rb") as f:
            similarity = pickle.load(f)
        return music_df, similarity
    except:
        return pd.DataFrame({'song': [], 'artist': []}), None

with st.spinner("Loading models and dataset..."):
    music, similarity = load_models()

if similarity is None:
    st.error("Model files not found or failed to load.")
    st.stop()

#GET SONG DETAILS:--
@st.cache_data()
def get_song_details(song_name: str, artist_name: str) -> Optional[Dict[str, Any]]:
    try:
        query = f"track:{song_name} artist:{artist_name}"
        results = sp.search(q=query, type="track", limit=1)
        items = results.get("tracks", {}).get("items", [])
        if not items:
            return None

        track = items[0]
        images = track.get("album", {}).get("images", [])
        image_url = images[0]["url"] if images else ""
        api_artist = track.get("artists", [{}])[0].get("name", artist_name)

        return {
            "image": image_url,
            "preview": track.get("preview_url"),
            "link": track.get("external_urls", {}).get("spotify"),
            "artist": api_artist,
            "album": track.get("album", {}).get("name", "Unknown album")
        }
    except:
        return None

#RECOMMENDATION FUNCTION:--
def recommend(song: str, top_k: int = 15, hide_no_preview: bool = False,
              year_range: tuple = (1950, 2025), mood: str = "Any"):

    index_list = music[music["song"] == song].index
    if len(index_list) == 0:
        return [], []

    index = index_list[0]

    try:
        row = similarity[index]
    except:
        return [], []

    distances = sorted(list(enumerate(row)), reverse=True, key=lambda x: x[1])

    names = []
    details = []

    st.sidebar.info(f"Using **{mood}** mood and year range **{year_range[0]}-{year_range[1]}** (display only).")

    for i, _ in distances[1:]:
        if len(names) >= top_k:
            break

        artist = music.iloc[i].artist
        song_name = music.iloc[i].song

        d = get_song_details(song_name, artist)
        if d is None:
            continue
        if hide_no_preview and not d["preview"]:
            continue

        names.append(song_name)
        details.append(d)

    return names, details

#PAGE HEADER:--
st.markdown("""
<h1 style='text-align:center; color:#00ffea;'>
    <i class='fa-brands fa-spotify'></i> AI Music Recommender
</h1>
<p style='text-align:center; font-size:18px; color:#d1d1d1;'>
    Discover your next favourite song. Recommendations powered by ML similarity and enriched with Spotify data.
</p>
""", unsafe_allow_html=True)
st.markdown("---")

#SIDEBAR FILTERS:---
with st.sidebar:
    st.subheader("1. Filter Dataset")

    all_artists = sorted(music["artist"].dropna().unique().tolist()) if not music.empty else []
    artist_options = ["All Artists"] + all_artists
    selected_artist = st.selectbox("Select your artist", artist_options, index=0)

    # ----------------- FIXED MASK (final solution) -----------------
    if not music.empty:
        mask = pd.Series(True, index=music.index)
    else:
        mask = pd.Series([], dtype=bool)

    if selected_artist != "All Artists":
        mask = mask & (music["artist"] == selected_artist)

    filtered_music = music[mask].sort_values(by="song") if not music.empty else music

    if filtered_music.empty and not music.empty:
        st.warning("No songs match this search. Showing full list.")
        filtered_music = music.sort_values(by="song")

    st.subheader("2. Select Seed Song")
    seed_song_options = filtered_music["song"].unique().tolist() if not filtered_music.empty else []

    if not seed_song_options:
        st.error("No songs available to select.")
        st.stop()

    seed_song = st.selectbox("Select the song to get recommendations for:", seed_song_options)

    if seed_song:
        seed_track_row = music[music["song"] == seed_song].iloc[0]
        seed_artist_name = seed_track_row["artist"]
        seed_details = get_song_details(seed_song, seed_artist_name)
        if seed_details:
            st.markdown(f"""
            <div class="seed-song-container">
                <img src="{seed_details['image']}" class="seed-song-image" alt="Seed Cover" />
                <div class="seed-song-info">
                    <p class="seed-song-title">{seed_song}</p>
                    <p class="seed-song-artist">by {seed_details['artist']}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info(f"Seed: **{seed_song}** by *{seed_artist_name}*")

    st.subheader("3. Recommendation Settings")
    num_recs = st.slider("Number of recommendations", min_value=5, max_value=25, value=12, step=1)
    hide_no_preview = st.checkbox("Hide tracks without Spotify preview", value=False)

    with st.expander("Advanced Recommendation Options"):
        year_min, year_max = st.slider("Year Range (Model Dependent)", min_value=1950, max_value=2025, value=(1970, 2025))
        mood = st.select_slider("Mood (Model Dependent)", options=["Any", "Chill", "Happy", "Energetic", "Relaxing"], value="Any")

    st.markdown("---")
    generate_btn = st.button("Generate Recommendations 🎧")

#TABS :--------
tab_recs, tab_about = st.tabs(["Recommendations", "About & Stats"])

with tab_recs:
    if generate_btn and seed_song:
        with st.spinner(f"Finding {num_recs} similar tracks for {seed_song}..."):
            names, details = recommend(
                seed_song,
                top_k=num_recs,
                hide_no_preview=hide_no_preview,
                year_range=(year_min, year_max),
                mood=mood
            )

        if not names:
            st.error("No recommendations found. Try another seed song or relax filters.")
        else:
            st.markdown(f"## Top {len(names)} Recommendations for: **{seed_song}**")
            st.markdown("---")

            cards_per_row = 4
            for row_start in range(0, len(names), cards_per_row):
                cols = st.columns(cards_per_row)
                for col_idx, col in enumerate(cols):
                    i = row_start + col_idx
                    if i >= len(names):
                        break

                    track_name = names[i]
                    d = details[i]

                    with col:
                        has_preview = d.get("preview")
                        preview_tag = ("<span class='song-tag'>Preview</span>" if has_preview else "<span class='song-tag'>No Preview</span>")
                        preview_label_class = "song-preview-label ok" if has_preview else "song-preview-label"
                        preview_label_text = "▶ Preview available" if has_preview else "⏹ No preview"

                        if d.get("link"):
                            spotify_btn_html = (
                                f"<a href='{d['link']}' target='_blank' class='song-spotify-btn' title='Open in Spotify'>"
                                f"<i class='fa-brands fa-spotify'></i><span>Spotify</span></a>"
                            )
                        else:
                            spotify_btn_html = "<span></span>"

                        card_html = f"""
<div class="song-card">
    <img src="{d.get('image', '')}" class="song-cover" alt="Album Cover" />
    <p class="song-title">{track_name}</p>
    <p class="song-artist">{d.get('artist', 'Unknown Artist')}</p>
    <p class="song-album">{d.get('album', 'Unknown Album')}</p>
    <div class="song-tags">
        <span class="song-tag">AI Match</span>
        {preview_tag}
    </div>
    <div class="song-actions">
        <span class="{preview_label_class}">{preview_label_text}</span>
        {spotify_btn_html}
    </div>
</div>
"""
                        st.markdown(card_html, unsafe_allow_html=True)

                        if has_preview:
                            st.audio(d["preview"], loop=False)

    else:
        st.info("Use the sidebar to configure settings and click Generate Recommendations 🎧.")

with tab_about:
    st.subheader("App Overview & Quick Stats")
    unique_songs = music["song"].nunique() if not music.empty else 0

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Entries in Dataset", len(music))
    with col2:
        st.metric("Unique Songs in Dataset", unique_songs)
    with col3:
        st.metric("Unique Artists in Dataset", len(all_artists))
    with col4:
        st.metric("Max Recommendations per Run", num_recs)
    
    st.markdown("---")
    st.markdown("#### How it works")         

    st.write(
        "- The app uses a precomputed similarity matrix to instantly find songs most similar to the track you choose.\n"
        "- Live metadata, album art, audio previews, and links are fetched using the Spotify Web API.\n"
        "- Smart filters, organized tabs, and custom‑styled cards create a smooth and engaging music‑discovery experience."
    )
    st.markdown("#### Created by Rajeev Kumar 🎵")

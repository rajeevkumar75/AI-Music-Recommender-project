import os
import pickle
import streamlit as st
import spotipy
import faiss
import numpy as np
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Music Recommender",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Spotify Credentials ---
# CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
# CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")

# For Streamlit Cloud secrets
CLIENT_ID = st.secrets["SPOTIFY_CLIENT_ID"]
CLIENT_SECRET = st.secrets["SPOTIFY_CLIENT_SECRET"]

if not CLIENT_ID or not CLIENT_SECRET:
    st.error("Missing Spotify API Credentials.")
    st.stop()

sp = spotipy.Spotify(
    client_credentials_manager=SpotifyClientCredentials(
        client_id=CLIENT_ID, client_secret=CLIENT_SECRET
    )
)

# --- Advanced UI/UX Styling ---
st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">
<style>
    /* Global Background and Font */
    .stApp {
        background: radial-gradient(circle at top right, #1db95420, #121212 40%);
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #000000;
        border-right: 1px solid #333;
    }

    /* Main Title Styling */
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #1DB954, #00ffea);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }

    /* Glassmorphism Song Card */
    .song-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 20px;
        margin-bottom: 30px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        min-height: 520px; /* Prevents overlapping players */
        display: flex;
        flex-direction: column;
        transition: all 0.3s ease-in-out;
    }
    
    .song-card:hover {
        transform: translateY(-10px);
        border-color: #1DB954;
        box-shadow: 0 10px 30px rgba(29, 185, 84, 0.2);
        background: rgba(255, 255, 255, 0.08);
    }
    
    .song-cover {
        border-radius: 12px;
        width: 100%;
        aspect-ratio: 1/1;
        object-fit: cover;
        box-shadow: 0 8px 15px rgba(0,0,0,0.5);
    }
    
    .song-info {
        margin: 15px 0;
        text-align: center;
        flex-grow: 1;
    }
    
    .song-title {
        font-weight: 700;
        color: #ffffff;
        font-size: 1.1rem;
        display: -webkit-box;
        -webkit-line-clamp: 1;
        -webkit-box-orient: vertical;
        overflow: hidden;
    }
    
    .song-artist {
        font-size: 0.9rem;
        color: #1DB954;
        font-weight: 500;
        margin-top: 5px;
    }
    
    /* Player Container */
    .spotify-container {
        height: 80px;
        border-radius: 12px;
        overflow: hidden;
        margin-top: auto;
    }

    /* Styled Buttons */
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #1DB954, #19e68c);
        color: black !important;
        font-weight: 700;
        border: none;
        border-radius: 12px;
        padding: 0.75rem;
        transition: 0.3s;
    }
    .stButton>button:hover {
        box-shadow: 0 0 20px rgba(29, 185, 84, 0.6);
        transform: scale(1.02);
    }
</style>
""", unsafe_allow_html=True)

# --- Data Loading ---
@st.cache_resource
def load_models():
    with open("models/df.pkl", "rb") as f:
        music_df = pickle.load(f)
    index = faiss.read_index("models/music_faiss.index")
    embeddings = np.load("models/music_embeddings.npy")
    return music_df.reset_index(drop=True), index, embeddings

music, faiss_index, embeddings = load_models()

# --- Functions ---
@st.cache_data(show_spinner=False)
def get_song_details(song, artist):
    try:
        res = sp.search(q=f"track:{song} artist:{artist}", type="track", limit=1)
        item = res["tracks"]["items"][0]
        return {
            "id": item["id"],
            "image": item["album"]["images"][0]["url"],
            "artist": item["artists"][0]["name"]
        }
    except Exception:
        return None

def recommend(song_name, top_k):
    idx_list = music[music["song"] == song_name].index
    if len(idx_list) == 0: return []
    idx = idx_list[0]
    query = embeddings[idx].reshape(1, -1)
    faiss.normalize_L2(query)
    _, indices = faiss_index.search(query, top_k + 1)
    
    results = []
    for i in indices[0]:
        if i == idx: continue
        row = music.iloc[i]
        details = get_song_details(row.song, row.artist)
        if details: results.append((row.song, details))
        if len(results) == top_k: break
    return results

# --- Main UI ---
st.markdown("<h1 class='main-title'><i class='fa-brands fa-spotify'></i> AI Music Recommender</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#aaa; margin-bottom:40px;'>FAISS-powered real-time music recommendations with Spotify integration</p>", unsafe_allow_html=True)

tab_reco, tab_about = st.tabs(["🎧 Recommendations", "👨‍💻 About"])

with tab_reco:
    with st.sidebar:
        st.markdown("### 🔍 Configure Search")
        artists = sorted(music["artist"].dropna().unique().tolist())
        selected_artist = st.selectbox("1. Pick an Artist", ["All Artists"] + artists)
        
        filtered_music = music if selected_artist == "All Artists" else music[music["artist"] == selected_artist]
        seed_song = st.selectbox("2. Select Reference Song", filtered_music["song"].sort_values().unique().tolist())

        st.markdown("---")
        num_recs = st.slider("Total Results", 3, 21, 12, step=3)
        generate = st.button("Generate Recommendation ✨")

    if generate and seed_song:
        with st.spinner("Analyzing audio vectors..."):
            results = recommend(seed_song, num_recs)
            
            # 3-Column Grid for optimal spacing
            for i in range(0, len(results), 3):
                cols = st.columns(3)
                for j in range(3):
                    if i + j < len(results):
                        name, d = results[i + j]
                        with cols[j]:
                            st.markdown(f"""
                            <div class="song-card">
                                <img src="{d['image']}" class="song-cover">
                                <div class="song-info">
                                    <div class="song-title">{name}</div>
                                    <div class="song-artist">{d['artist']}</div>
                                </div>
                                <div class="spotify-container">
                                    <iframe 
                                        src="https://open.spotify.com/embed/track/{d['id']}?utm_source=generator&theme=0" 
                                        width="100%" height="80" frameBorder="0" 
                                        allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture" 
                                        loading="lazy"></iframe>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
    else:
        st.info("Start by choosing a song from the sidebar to find your next favorite track!")

with tab_about:
    st.markdown("""
    ### How This work:-
    This recommendation engine uses **High-Dimensional Embeddings** to understand the "soul" of a song. Unlike simple genre filters, it analyzes audio features to find tracks with similar moods, tempos, and textures.
    
    - **Search Engine:** FAISS (Facebook AI Similarity Search)
    - **API:** Spotify Web API
    - **Scale:** Real-time retrieval across 15,000+ tracks
    """)

    st.markdown("---")
    st.info(" © 20260 Rajeev Kumar. All rights reserved. ", icon="⚖️")

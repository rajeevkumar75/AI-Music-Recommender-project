import os
import pickle
import streamlit as st
import spotipy
import faiss
import numpy as np
import pandas as pd
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv
from datetime import datetime
import csv
from io import StringIO

load_dotenv()

st.set_page_config(page_title="AI Music Recommender 🎧", page_icon="🎵", layout="wide")
st.markdown('<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">', unsafe_allow_html=True)

st.markdown("""
<style>
.song-card { 
  background: linear-gradient(135deg, #1a1a1a 0%, #0d2828 100%); 
  border-radius: 20px; 
  padding: 16px; 
  text-align: center; 
  box-shadow: 0 8px 32px rgba(0, 255, 200, 0.15), inset 0 1px 0 rgba(255, 255, 255, 0.1); 
  border: 1px solid rgba(0, 255, 200, 0.3); 
  margin-bottom: 16px;
  transition: all 0.3s ease;
  position: relative;
  overflow: hidden;
}
.song-card:hover { 
  transform: translateY(-8px);
  box-shadow: 0 12px 40px rgba(0, 255, 200, 0.25), inset 0 1px 0 rgba(255, 255, 255, 0.1);
  border-color: rgba(0, 255, 200, 0.6);
}
.song-card::before {
  content: '';
  position: absolute;
  top: 0;
  left: -100%;
  width: 100%;
  height: 100%;
  background: linear-gradient(90deg, transparent, rgba(0, 255, 200, 0.1), transparent);
  transition: left 0.5s ease;
}
.song-card:hover::before {
  left: 100%;
}
.song-cover { 
  border-radius: 14px; 
  width: 100%; 
  aspect-ratio: 1/1; 
  object-fit: cover;
  border: 2px solid rgba(0, 255, 200, 0.2);
  transition: transform 0.3s ease;
}
.song-card:hover .song-cover {
  transform: scale(1.05);
}
.song-title { 
  font-weight: 700; 
  color: #fff; 
  margin: 12px 0 6px 0;
  line-height: 1.3;
  font-size: 15px;
}
.song-artist { 
  font-size: 13px; 
  color: #00ffc8; 
  font-weight: 600;
  margin: 4px 0;
}
.song-album { 
  font-size: 11px; 
  color: #888; 
  margin: 4px 0;
}
.song-duration {
  font-size: 10px;
  color: #666;
  margin: 6px 0;
  font-weight: 500;
}
.song-buttons-container {
  display: flex;
  gap: 8px;
  margin-top: 12px;
  justify-content: center;
  flex-wrap: wrap;
}
.song-spotify-btn { 
  padding: 8px 14px; 
  border-radius: 999px; 
  background: linear-gradient(135deg, #1db954 0%, #1ed760 100%);
  color: white; 
  font-size: 12px; 
  font-weight: bold; 
  text-decoration: none;
  border: none;
  cursor: pointer;
  transition: all 0.3s ease;
  display: inline-flex;
  align-items: center;
  gap: 6px;
  flex: 1;
  justify-content: center;
}
.song-spotify-btn:hover { 
  background: linear-gradient(135deg, #1ed760 0%, #1db954 100%);
  transform: scale(1.05);
  box-shadow: 0 4px 12px rgba(30, 215, 96, 0.3);
}
.seed-container { 
  display: flex; 
  gap: 12px; 
  align-items: center; 
  padding: 12px 14px; 
  border-radius: 12px; 
  background: linear-gradient(135deg, #1f1f1f 0%, #0d2828 100%); 
  border: 1px solid rgba(0, 255, 200, 0.3);
  margin-top: 10px;
  box-shadow: 0 4px 12px rgba(0, 255, 200, 0.1);
}
.seed-img { 
  width: 60px; 
  height: 60px;     
  border-radius: 8px; 
  object-fit: cover;
  border: 2px solid rgba(0, 255, 200, 0.4);
}
.stat-card { 
  background: linear-gradient(135deg, #00ffc8 0%, #00ffe5 100%); 
  border-radius: 12px; 
  padding: 20px; 
  margin: 10px 0; 
  color: #111; 
  font-weight: bold; 
  text-align: center;
  box-shadow: 0 8px 24px rgba(0, 255, 200, 0.2);
  transition: transform 0.3s ease;
}
.stat-card:hover {
  transform: translateY(-4px);
}
.tech-badge { 
  display: inline-block; 
  background: linear-gradient(135deg, #00ffc8 0%, #00ffe5 100%); 
  color: #111; 
  padding: 6px 14px; 
  border-radius: 20px; 
  font-size: 12px; 
  font-weight: bold; 
  margin: 5px 5px 5px 0;
  box-shadow: 0 2px 8px rgba(0, 255, 200, 0.15);
}
</style>
""", unsafe_allow_html=True)

# Spotify Setup
CLIENT_ID = st.secrets["SPOTIFY_CLIENT_ID"]
CLIENT_SECRET = st.secrets["SPOTIFY_CLIENT_SECRET"]
if not CLIENT_ID or not CLIENT_SECRET:
    st.error("⚠️ Set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET environment variables")
    st.stop()

sp = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET))

# Load Models
@st.cache_resource
def load_models():
    music_df = pickle.load(open("models/df.pkl", "rb"))
    index = faiss.read_index("models/music_faiss.index")
    embeddings = np.load("models/music_embeddings.npy")
    return music_df.reset_index(drop=True), index, embeddings

music, faiss_index, embeddings = load_models()

# Cache for Spotify details
if 'spotify_cache' not in st.session_state:
    st.session_state.spotify_cache = {}

def get_song_details(song, artist):
    cache_key = f"{song}|{artist}"
    if cache_key in st.session_state.spotify_cache:
        return st.session_state.spotify_cache[cache_key]
    
    try:
        res = sp.search(q=f"track:{song} artist:{artist}", type="track", limit=1)
        if not res["tracks"]["items"]:
            st.session_state.spotify_cache[cache_key] = None
            return None
        item = res["tracks"]["items"][0]
        result = {
            "image": item["album"]["images"][0]["url"] if item["album"]["images"] else "",
            "preview": item["preview_url"],
            "link": item["external_urls"]["spotify"],
            "artist": item["artists"][0]["name"],
            "album": item["album"]["name"],
            "release_date": item["album"].get("release_date", "Unknown"),
            "duration_ms": item["duration_ms"]
        }
        st.session_state.spotify_cache[cache_key] = result
        return result
    except:
        st.session_state.spotify_cache[cache_key] = None
        return None

def format_duration(ms):
    return f"{ms // 60000}:{(ms % 60000) // 1000:02d}"

def recommend(song, top_k, hide_no_preview):
    idx_list = music[music["song"] == song].index
    if len(idx_list) == 0:
        return []
    
    idx = idx_list[0]
    query = embeddings[idx].reshape(1, -1)
    faiss.normalize_L2(query)
    
    _, indices = faiss_index.search(query, min(top_k * 2 + 5, len(embeddings)))
    
    results = []
    for i in indices[0]:
        if i == idx:
            continue
        row = music.iloc[i]
        d = get_song_details(row.song, row.artist)
        if not d or (hide_no_preview and not d.get("preview")):
            continue
        results.append((row.song, d))
        if len(results) == top_k:
            break
    return results

def create_playlist_text(song_list, seed_song):
    playlist = f"🎵 Playlist: {seed_song}\n📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n🎧 Songs: {len(song_list)}\n" + "-"*50 + "\n\n"
    for i, (song, d) in enumerate(song_list, 1):
        playlist += f"{i}. {song} - {d['artist']}\n   Album: {d['album']}\n   Duration: {format_duration(d['duration_ms'])}\n   Link: {d['link']}\n\n"
    return playlist

def create_playlist_csv(song_list, seed_song):
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(["Song", "Artist", "Album", "Duration", "Spotify Link"])
    for song, d in song_list:
        writer.writerow([song, d['artist'], d['album'], format_duration(d['duration_ms']), d['link']])
    return output.getvalue()

def get_artist_stats(artist_name):
    artist_songs = music[music["artist"] == artist_name]
    return {
        "songs": len(artist_songs),
        "avg_popularity": artist_songs["popularity"].mean() if "popularity" in artist_songs.columns else 0,
        "genres": artist_songs["genre"].unique().tolist() if "genre" in artist_songs.columns else []
    }

# UI
st.markdown("<h1 style='text-align:center; color:#00ffea;'><i class='fa-brands fa-spotify'></i> AI Music Recommender</h1>", unsafe_allow_html=True)
st.markdown("---")

tab_reco, tab_stats, tab_about = st.tabs(["🎧 Recommendations", "📊 Stats", "ℹ️ About"])

# RECOMMENDATIONS TAB
with tab_reco:
    with st.sidebar:
        st.markdown("### 🎵 Select Seed Song")
        artists = sorted(music["artist"].dropna().unique().tolist())
        selected_artist = st.selectbox("Filter by Artist", ["All Artists"] + artists)
        filtered_music = music if selected_artist == "All Artists" else music[music["artist"] == selected_artist]
        
        seed_song = st.selectbox("Select Song", filtered_music["song"].sort_values().unique().tolist())
        
        if seed_song:
            seed_row = music[music["song"] == seed_song].iloc[0]
            seed_details = get_song_details(seed_song, seed_row.artist)
            if seed_details:
                st.markdown(f"""<div class="seed-container">
                    <img src="{seed_details['image']}" class="seed-img">
                    <div><b style="color:white">{seed_song}</b><br><span style="color:#b3b3b3">{seed_details['artist']}</span><br>
                    <span style="color:#888; font-size:11px;">{format_duration(seed_details['duration_ms'])}</span></div></div>""", unsafe_allow_html=True)
        
        hide_no_preview = st.checkbox("Hide songs without preview", value=False)
        num_recs = st.slider("Number of recommendations", 5, 25, 12)
        generate = st.button("🎧 Recommend", use_container_width=True)
    
    if generate and seed_song:
        with st.spinner("🔍 Finding tracks..."):
            results = recommend(seed_song, num_recs, hide_no_preview)
        
        if results:
            st.success(f"✅ Found {len(results)} recommendations!")
            
            # Display recommendations
            cols = st.columns(4)
            for i, (name, d) in enumerate(results):
                with cols[i % 4]:
                    img = f'<img src="{d["image"]}" class="song-cover">' if d.get("image") else '<div class="song-cover" style="background: linear-gradient(135deg, #333 0%, #111 100%); display: flex; align-items: center; justify-content: center; color: #888;">No Image</div>'
                    st.markdown(f"""<div class="song-card">{img}
                    <p class="song-title">{name}</p><p class="song-artist">{d['artist']}</p>
                    <p class="song-album">{d['album']}</p><p class="song-duration">{format_duration(d['duration_ms'])}</p>
                    <div class="song-buttons-container">
                    <a href="{d['link']}" target="_blank" class="song-spotify-btn" style="text-decoration:none; color:white; font-weight:bold; font-size:14px;"><i class="fa-brands fa-spotify"></i> Spotify</a></div></div>""", unsafe_allow_html=True)
                    
                    if d.get("preview"):
                        st.audio(d["preview"], format="audio/mpeg")
            
            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                st.download_button("📥 Download Playlist (TXT)", data=create_playlist_text(results, seed_song), 
                                  file_name=f"playlist_{seed_song.replace(' ', '_')}.txt", mime="text/plain", use_container_width=True)
            with col2:
                st.download_button("📊 Download Playlist (CSV)", data=create_playlist_csv(results, seed_song),
                                  file_name=f"playlist_{seed_song.replace(' ', '_')}.csv", mime="text/csv", use_container_width=True)
        else:
            st.warning("⚠️ No results found. Try disabling preview filter.")
    elif not seed_song:
        st.info("👈 Select a song and click Generate")

# STATS TAB
with tab_stats:
    st.subheader("📊 Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="stat-card">🎵<br>{len(music):,}<br><span style="font-size: 12px;">Total Songs</span></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="stat-card">🎤<br>{music["artist"].nunique():,}<br><span style="font-size: 12px;">Artists</span></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="stat-card">🎸<br>{music["genre"].nunique() if "genre" in music.columns else 0}<br><span style="font-size: 12px;">Genres</span></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="stat-card">📊<br>{len(music.columns)}<br><span style="font-size: 12px;">Features</span></div>', unsafe_allow_html=True)
    
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Top 10 Artists**")
        st.bar_chart(music["artist"].value_counts().head(10))
    with col2:
        st.write("**Dataset Columns**")
        st.dataframe(pd.DataFrame({"Column": music.columns, "Type": [str(music[col].dtype) for col in music.columns]}), use_container_width=True, hide_index=True)
    
    st.markdown("---")
    with st.expander("🔎 Search Songs & Artists"):
        search_term = st.text_input("Enter song or artist name")
        if search_term:
            results = music[(music["song"].str.contains(search_term, case=False, na=False)) | (music["artist"].str.contains(search_term, case=False, na=False))].head(20)
            if len(results) > 0:
                st.success(f"Found {len(results)} results")
                st.dataframe(results[["song", "artist"]], use_container_width=True, hide_index=True)
            else:
                st.info("No results found")

# ABOUT TAB
with tab_about:
    st.subheader("🎵 AI Music Recommender")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
### ✨ Features
- ⚡ FAISS-powered instant search
- 🎧 Spotify integration (artwork & previews)
- 📊 Compare recommendations
- 📥 Export playlists (TXT/CSV)
- 🎵 High-quality music embeddings
""")
    
    with col2:
        st.markdown("""
### 📊 Dataset & Tech
- 57,000+ songs indexed
- Multiple audio features
- FAISS Vector Search
- Spotify Web API
- Streamlit Frontend
""")
    
    st.markdown("---")
    st.info("🎵 **Tip:** Use the Recommendations tab to find similar songs based on your selection! Download your favorite playlists in TXT or CSV format.")

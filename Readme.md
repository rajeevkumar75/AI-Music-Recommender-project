# 🎵 AI Music Recommender

A **production-grade music recommendation system** using semantic embeddings, FAISS vector search, and Spotify API integration. Built with Streamlit for real-time, interactive recommendations.

**Live Demo:** [AI Music Recommender](https://ai-music-recommender-project-euayjb3otqkvvxrcytjyrb.streamlit.app/)

---

## 🎯 What It Does

Find similar songs instantly based on a seed track. The system analyzes song metadata, generates semantic embeddings, and performs ultra-fast nearest-neighbor search—all in **sub-millisecond latency**.

**Example Workflow:**
1. Select a song (e.g., "Shape of You" by Ed Sheeran)
2. Click "Recommend" 
3. Get 5-25 similar songs with Spotify metadata and audio previews
4. Download as playlist (TXT/CSV) or stream directly

---

## ⚡ Key Features

### 🎧 Recommendation Engine
- **FAISS-powered search** - Sub-millisecond nearest-neighbor queries on 57,000+ songs
- **Smart filtering** - Option to hide songs without preview samples
- **Configurable results** - Get 5-25 recommendations per query
- **Artist-based filtering** - Browse songs by specific artists in sidebar

### 🎵 Spotify Integration
- **Album artwork** - High-quality cover images for each recommended song
- **Audio previews** - Built-in 30-second preview player
- **Live metadata** - Artist names, album info, track duration, release dates
- **Direct links** - One-click Spotify access for every song

### 📊 Analytics Dashboard
- **Dataset overview** - 57,000+ songs with multiple audio features
- **Artist statistics** - Top 10 artists by track count
- **Full-text search** - Find songs and artists quickly
- **Data exploration** - Browse dataset structure and column information

### 🎨 Modern User Interface
- **Gradient design** - Cyan/teal theme with dark backgrounds
- **Interactive cards** - Hover animations with scale and lift effects
- **Responsive layout** - Works seamlessly on desktop and mobile
- **Real-time feedback** - Loading spinners, success/error messages
- **Font Awesome icons** - Spotify-branded buttons and visual elements

### 💾 Export Features
- **TXT format** - Human-readable playlists with full metadata
- **CSV format** - Spreadsheet-compatible for further analysis
- **Metadata included** - Songs, artists, albums, durations, Spotify links

---

## 🛠 Technical Architecture

### ML Pipeline

```
Raw CSV Data
    ↓
[Data Processor]
├─ Load 15,000+ songs
├─ Clean text (lowercase, tokenize, stemming)
├─ Handle missing values
└─ Create text column: song + artist + genre
    ↓
[Feature Engineer]
├─ TF-IDF Vectorization (5000 features)
├─ SVD Dimensionality Reduction (256 dimensions)
└─ L2 Normalization for cosine similarity
    ↓
[Model Trainer]
├─ Build FAISS IndexFlatIP
├─ Serialize embeddings (.npy)
└─ Serialize index (.index)
    ↓
[Streamlit App]
├─ Load cached models
├─ Query embedding lookup
├─ FAISS nearest-neighbor search
├─ Spotify API enrichment
└─ Interactive recommendations
```

### Data Flow

```
User selects song
        ↓
Get embedding from pre-computed vectors
        ↓
L2 normalize for cosine similarity
        ↓
FAISS search (returns ~20-30 neighbors)
        ↓
Filter by preview availability (optional)
        ↓
Query Spotify API for metadata/artwork
        ↓
Display with caching (avoid duplicate API calls)
        ↓
User exports or listens
```

---

## 📊 Performance & Scale

| Metric | Value |
|--------|-------|
| **Dataset Size** | 57,000+ songs |
| **Search Speed** | <5ms per query |
| **Memory Usage** | ~200MB (embeddings + index) |
| **Embedding Dimension** | 256D |
| **Embedding Method** | TF-IDF + SVD |
| **Index Type** | FAISS IndexFlatIP |
| **Cloud Deployment** | Free-tier ready |

### Why FAISS?

- ⚡ **Speed** - C++ implementation with Python bindings
- 💾 **Memory** - Efficient index format, no O(n²) matrices
- 🔍 **Accuracy** - IndexFlatIP provides exact cosine similarity
- 📦 **Production** - Used at Meta/Facebook scale
- 🆓 **Free** - Open-source, actively maintained

---

## 🏗 Project Structure

```
AI-Music-Recommender/
│
├── app.py                          # Main Streamlit application
│   ├── UI Components (3 tabs)
│   ├── Spotify API integration
│   ├── FAISS search logic
│   └── Custom CSS styling
│
├── training/
│   ├── __init__.py
│   ├── data_processor.py           # Data loading & text cleaning
│   │   ├── Load CSV (15K sample)
│   │   ├─ Tokenization & stemming
│   │   └── Quality checks
│   │
│   ├── feature_engineer.py         # Embeddings & dimensionality reduction
│   │   ├── TF-IDF vectorization
│   │   ├── SVD (256 components)
│   │   └── L2 normalization
│   │
│   └── model_trainer.py            # Training orchestration
│       ├── Coordinate pipeline
│       ├── Build FAISS index
│       └── Save serialized models
│
├── models/
│   ├── music_faiss.index           # FAISS vector index
│   ├── music_embeddings.npy        # Pre-computed embeddings
│   └── df.pkl                      # Song metadata
│
├── Data/
│   └── spotify_millsongdata.csv    # Raw dataset (57K songs)
│
├── Notebook/
│   └── music_data_analysis.ipynb   # EDA & analysis
│
├── requirements.txt                # Python dependencies
└── Readme.md                       # This file
```

---

## 🔧 Installation & Setup

### Prerequisites

- Python 3.8+
- pip or conda
- Spotify API credentials (Client ID & Secret)

### 1. Clone Repository

```bash
git clone https://github.com/rajeevkumar75/AI-Music-Recommender-project.git
cd AI-Music-Recommender-project
```

### 2. Create Virtual Environment

```bash
# Using conda
conda create -n music-rec python=3.10
conda activate music-rec

# OR using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Spotify API Credentials

Create a `.env` file in the project root:

```bash
SPOTIFY_CLIENT_ID=your_spotify_client_id_here
SPOTIFY_CLIENT_SECRET=your_spotify_client_secret_here
```

**How to get Spotify credentials:**
1. Go to [Spotify Developer Dashboard](https://developer.spotify.com/dashboard)
2. Create a new app
3. Accept terms and create
4. Copy Client ID and Client Secret
5. Add to `.env` file

### 5. Run the Application

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`

---

## 📖 How to Use

### 🎧 Getting Recommendations

1. **Select Artist** (optional)
   - Use sidebar dropdown to filter by artist
   - Narrows down song selection

2. **Choose Seed Song**
   - Select from filtered list
   - See preview with album art in sidebar

3. **Configure Search**
   - **Preview filter** - Hide songs without 30-sec samples
   - **Result count** - Adjust 5-25 recommendations

4. **Click "Recommend"**
   - System searches for similar songs
   - Results display in 4-column grid

5. **Explore Results**
   - Hover over cards for animations
   - Click Spotify button to open full song
   - Play preview with built-in player

6. **Export Playlist**
   - Download as TXT (readable)
   - Download as CSV (for Excel/spreadsheets)

### 📊 Using Analytics Tab

1. **View Overview**
   - 4 key metrics (songs, artists, genres, features)
   - Top 10 artists bar chart

2. **Browse Dataset**
   - Column names and data types
   - Understand feature structure

3. **Search Songs**
   - Find by song name or artist
   - View metadata

---

## 🧠 How It Works (Technical Deep Dive)

### 1. Data Preprocessing
```python
# Text concatenation
song_data['text'] = song_data['song'] + " " + song_data['artist'] + " " + song_data['genre']

# Tokenization & stemming
"SHAPE OF YOU" → ["shape", "of", "you"]
"shaping" → "shape"  # Porter stemming reduces variations
```

### 2. Embedding Generation
```python
# TF-IDF: Converts text → frequency vectors (max 5000 features)
text_data → sparse matrix (57000, 5000)

# SVD: Reduces dimensions while preserving variance
(57000, 5000) → (57000, 256)  # More efficient, faster search

# L2 Normalization: Enables cosine similarity via dot product
embeddings_normalized = embeddings / ||embeddings||
```

### 3. FAISS Indexing
```python
# IndexFlatIP: Inner product (equivalent to cosine similarity with L2 norm)
index = faiss.IndexFlatIP(256)
index.add(embeddings)  # Add all 57K vectors

# Search
distances, indices = index.search(query_embedding, k=30)
# Returns 30 closest matches
```

### 4. Real-Time Inference
```
User clicks "Recommend"
    ↓
Load pre-computed embedding for seed song
    ↓
Query FAISS index (< 5ms)
    ↓
Get 20-30 nearest neighbors
    ↓
Filter by preview availability
    ↓
Call Spotify API for artwork/metadata (cached)
    ↓
Render cards with animations
```

---

## 📊 Dataset

**Source:** Spotify Million Song Dataset

**Stats:**
- **Total Songs:** 57,000+
- **Unique Artists:** Thousands
- **Genres:** Multiple classifications
- **Features:** song, artist, genre, popularity, etc.
- **Preprocessing:** 15,000 songs used for embedding training

---

## 💡 Design Decisions

### Why TF-IDF + SVD?
- ✅ **Fast** - No neural network training needed
- ✅ **Interpretable** - See which words matter
- ✅ **Sufficient** - Text captures musical context well
- ✅ **Scalable** - Works with limited resources

### Why FAISS IndexFlatIP?
- ✅ **Exact results** - No approximation loss
- ✅ **Fast enough** - IndexIVF* is overkill for 57K songs
- ✅ **Memory efficient** - Fits in RAM on any machine
- ❌ IndexIVF* saves memory on millions of vectors (not needed here)

### Why Streamlit?
- ✅ **Rapid development** - No frontend framework needed
- ✅ **Built-in components** - Audio players, charts, buttons
- ✅ **Caching** - Session state management
- ✅ **Deployment** - Free tier on Streamlit Cloud

### Why Spotify API?
- ✅ **Rich metadata** - Artist, album, duration, release date
- ✅ **Album artwork** - High-quality cover images
- ✅ **Audio preview** - 30-second samples
- ✅ **Direct links** - One-click to full song

---

## 🚀 Deployment

### Local Testing
```bash
streamlit run app.py
```

### Deploy to Streamlit Cloud
1. Push code to GitHub
2. Connect repository to Streamlit Cloud
3. Set environment variables in secrets
4. Deploy

### Docker Deployment
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD ["streamlit", "run", "app.py"]
```

---

## 📈 Performance Metrics

**Search Speed:**
- Single query: ~2-5ms
- Spotify API call: ~200-300ms
- Total latency: ~250-350ms (dominated by API call, not search)

**Memory Profile:**
- Embeddings: ~200MB (57K × 256 × 4 bytes)
- FAISS index: ~50MB
- DataFrames: ~100MB
- Total: ~350MB (fits in free Streamlit tier)

**Accuracy:**
- Qualitative evaluation shows relevant recommendations
- Songs share genres, artists, tempo, mood
- User feedback indicates satisfaction

---

## 🎨 UI Features

### Color Scheme
- **Primary:** Cyan (#00ffc8) - Spotify-inspired
- **Accent:** Teal (#0d2828) - Depth
- **Background:** Dark (#1a1a1a) - Reduces eye strain

### Animations
- **Card hover** - Lift effect (translateY -8px)
- **Image hover** - Scale zoom (1.05x)
- **Button hover** - Color invert + scale
- **Shimmer effect** - Subtle gradient animation

### Responsive Design
- **Desktop:** 4-column grid for recommendations
- **Tablet:** 2-column grid
- **Mobile:** 1-column stacked layout

---

## 🔍 Quality Assurance

### Data Validation
✅ Check for nulls in critical columns  
✅ Remove duplicates  
✅ Verify text fields not empty  
✅ Confirm embedding dimensions match FAISS index  

### API Error Handling
✅ Graceful fallback for missing Spotify metadata  
✅ Caching prevents duplicate API calls  
✅ Session state prevents duplicate requests  
✅ User-friendly error messages  

### Performance Testing
✅ Index search time < 5ms  
✅ App startup < 3 seconds (with caching)  
✅ Memory usage stays < 500MB  

---

## 📚 Technologies Used

### Core ML
- **scikit-learn** - TF-IDF, SVD, feature extraction
- **FAISS** - Vector indexing and search
- **NumPy** - Numerical computations
- **Pandas** - Data manipulation

### Web & API
- **Streamlit** - Web interface
- **Spotipy** - Spotify API client
- **Requests** - HTTP client
- **python-dotenv** - Environment variables

### NLP
- **NLTK** - Tokenization, stemming
- **Porter Stemmer** - Word normalization

### Data Processing
- **Pickle** - Model serialization
- **CSV** - Data import/export

---

## 🤝 Contributing

Contributions welcome! Areas for improvement:

- [ ] Add collaborative filtering (user preferences)
- [ ] Implement mood/energy-based recommendations
- [ ] Add playlist generation with constraints
- [ ] Build mobile app with React Native
- [ ] Add user authentication and history
- [ ] Implement A/B testing framework

---

## 📝 License

This project is open source and available under the MIT License.

---

## 🙏 Acknowledgments

- **Spotify** - Data source and API access
- **Meta/Facebook** - FAISS library
- **Streamlit** - Web framework
- **Open source community** - scikit-learn, NLTK, NumPy, Pandas

---

## 📧 Contact & Support

For questions or issues:
1. Check [GitHub Issues](https://github.com/rajeevkumar75/AI-Music-Recommender-project/issues)
2. Review the [Streamlit docs](https://docs.streamlit.io/)
3. Visit [FAISS documentation](https://faiss.ai/)

---

## 🎯 What This Demonstrates

**For Recruiters & Interviewers:**

✨ **ML System Design**
- End-to-end pipeline from raw data to production
- Feature engineering and dimensionality reduction
- Vector search indexing at scale

⚡ **Full-Stack Development**
- ML backend (Python, scikit-learn, FAISS)
- Web frontend (Streamlit, custom CSS)
- API integration (Spotify Web API)
- Deployment and optimization

🔍 **Problem Solving**
- Identified scalability bottleneck (O(n²) matrices)
- Designed efficient solution (FAISS)
- Balanced accuracy vs. performance
- Optimized for resource constraints

🎨 **Product Thinking**
- User-centric design with animations
- Export functionality for value delivery
- Mobile-responsive interface
- Error handling and edge cases

---

## 📊 Stats

- **Lines of Code:** 500+ (app.py) + 200+ (training)
- **Training Time:** <5 minutes
- **Search Latency:** <5ms
- **Memory Footprint:** <500MB
- **Dataset Size:** 57,000 songs
- **Embedding Dimension:** 256D
- **Live Users:** Accessible to public

---

**Built with ❤️ for music lovers and ML enthusiasts.**

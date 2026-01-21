# 🎵 AI Music Recommender  

Streamlit web app: https://ai-music-recommender-project-euayjb3otqkvvxrcytjyrb.streamlit.app/

### 📌 Problem
Traditional content-based music recommendation systems rely on precomputed **cosine similarity matrices**, which do not scale well in terms of **memory and latency**. For a dataset of **15,000+ songs**, the similarity matrix alone consumed **~1.6 GB**, making it unsuitable for **real-time inference** and **free-tier cloud deployment**.

### 💡 Solution
The recommendation pipeline was redesigned using **semantic text embeddings** combined with **FAISS vector indexing**, eliminating the need for a full similarity matrix. Songs are represented as dense embedding vectors, and recommendations are generated using **Approximate Nearest Neighbor (ANN) search**, enabling **fast and memory-efficient retrieval**.

### 🧠 System Design
- **Data Processing:** NLP-based cleaning and normalization of song metadata  
- **Feature Representation:** Conversion of text into dense semantic embeddings  
- **Similarity Search:** FAISS-based vector indexing for low-latency nearest-neighbor queries  
- **Inference Pipeline:** Serialized embeddings and FAISS index for fast startup and real-time predictions  
- **Frontend:** Interactive Streamlit app integrated with the **Spotify Web API** for live metadata, album art, and previews  

### 📈 Scale & Performance
- **Dataset Size:** 15,000+ songs  
- **Memory Optimization:** Reduced storage from **~1.6 GB cosine matrix → lightweight FAISS index**  
- **Latency:** Near real-time Top-N recommendations  
- **Deployment:** Optimized for **free-tier cloud environments**

### ⚙️ Key Engineering Decisions
- Replaced brute-force cosine similarity with **ANN search** for scalability  
- Chose **FAISS** for speed, memory efficiency, and production adoption  
- Modularized the pipeline to separate **offline embedding generation** from **online inference**  
- Prioritized **low latency and cost efficiency** over increased model complexity  

### 🚀 Impact
- Enabled **real-time, scalable music recommendations** with minimal memory overhead  
- Demonstrated **production-oriented ML system design**, not just model training  
- Delivered a **deployable, user-facing application** aligned with industry best practices  

### 🛠 Tech Stack
**Python | Machine Learning | NLP | Text Embeddings | FAISS | Scikit-learn | Streamlit | Spotify Web API**

### 🎯 What This Demonstrates
- End-to-end **ML system design**
- **Scalability and performance optimization**
- Real-world use of **vector search and embeddings**
- Ability to build **production-ready ML applications**

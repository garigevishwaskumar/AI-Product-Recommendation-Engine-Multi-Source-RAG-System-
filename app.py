import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from collections import Counter

# ------------------------------
# Load Dataset
# ------------------------------
df = pd.read_csv("data.csv")

# Combine review + specs for RAG knowledge
df["text"] = df["product"] + " | " + df["specs"] + " | Review: " + df["review"]

# ------------------------------
# Load Embedding Model
# ------------------------------
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Generate embeddings
embeddings = embed_model.encode(df["text"].tolist(), convert_to_numpy=True)

# ------------------------------
# Build FAISS Index
# ------------------------------
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# ------------------------------
# Sentiment Model
# ------------------------------
sentiment_model = pipeline("sentiment-analysis")

# ------------------------------
# Helper Functions
# ------------------------------
def retrieve_docs(query, top_k=5):
    q_emb = embed_model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(q_emb, top_k)

    results = df.iloc[indices[0]].copy()
    results["distance"] = distances[0]
    return results


def analyze_complaints(reviews):
    negative_keywords = ["heating", "battery", "fan", "noise", "build", "speaker", "slow", "lag"]
    found = []

    for r in reviews:
        lower = r.lower()
        for kw in negative_keywords:
            if kw in lower:
                found.append(kw)

    counts = Counter(found)
    return counts


def rank_products(retrieved_df):
    grouped = retrieved_df.groupby("product")

    scores = []
    for product, group in grouped:
        reviews = group["review"].tolist()

        sentiments = sentiment_model(reviews)
        pos = sum(1 for s in sentiments if s["label"] == "POSITIVE")
        neg = sum(1 for s in sentiments if s["label"] == "NEGATIVE")

        complaints = analyze_complaints(reviews)
        complaint_penalty = sum(complaints.values()) * 0.5

        base_score = (pos * 2) - (neg * 2) - complaint_penalty

        scores.append({
            "product": product,
            "price": group["price"].iloc[0],
            "specs": group["specs"].iloc[0],
            "positive_reviews": pos,
            "negative_reviews": neg,
            "complaints_found": dict(complaints),
            "final_score": base_score
        })

    score_df = pd.DataFrame(scores).sort_values(by="final_score", ascending=False)
    return score_df


def generate_final_answer(query, ranked_df, evidence_df):
    best = ranked_df.iloc[0]

    evidence_text = "\n".join(
        [f"- {row['review']}" for _, row in evidence_df[evidence_df["product"] == best["product"]].head(3).iterrows()]
    )

    answer = f"""
### ✅ Best Recommendation: **{best['product']}**
💰 Price: ₹{best['price']}  
🧾 Specs: {best['specs']}

### ⭐ Why this is best for you:
- It has strong positive feedback from real reviews.
- It matches your query: **{query}**
- It has fewer negative complaints compared to others.

### ⚠️ Common Issues Found:
{best['complaints_found'] if best['complaints_found'] else "No major complaints detected from retrieved reviews."}

### 🔍 Evidence from Reviews:
{evidence_text}

### 📌 Final AI Score:
**{best['final_score']}**
"""
    return answer


# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="AI Product Buying Assistant", layout="wide")

st.title("🛒 AI Product Buying Assistant (RAG + Reviews + Sentiment)")
st.write("Ask like: *Best laptop under 60k for coding + ML with good battery and no heating*")

query = st.text_input("Enter your requirement:")

if st.button("Find Best Product"):
    if query.strip() == "":
        st.warning("Please enter a query.")
    else:
        st.info("🔍 Retrieving best matching reviews...")

        retrieved = retrieve_docs(query, top_k=8)
        ranked = rank_products(retrieved)

        st.subheader("🏆 Ranked Products")
        st.dataframe(ranked)

        st.subheader("🤖 Final AI Recommendation")
        final_answer = generate_final_answer(query, ranked, retrieved)
        st.markdown(final_answer)

        st.subheader("📌 Retrieved Evidence (RAG Docs)")
        st.dataframe(retrieved[["product", "review", "specs"]])
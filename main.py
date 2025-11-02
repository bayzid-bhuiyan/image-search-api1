from fastapi import FastAPI, File, UploadFile, HTTPException
import numpy as np
import pandas as pd
from PIL import Image
from sentence_transformers import SentenceTransformer
import faiss
import io
import os

app = FastAPI(title="Ecommerce Image Search API")

# ---- Paths ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "products_with_embeddings.csv")
EMB_PATH = os.path.join(BASE_DIR, "image_embeddings.npy")
IMAGE_URL_BASE = "https://ladonna.com.bd/image/uploads/"  # Change if hosting images elsewhere

# ---- Load CSV and embeddings ----
if not os.path.exists(CSV_PATH) or not os.path.exists(EMB_PATH):
    raise FileNotFoundError("CSV or embeddings file not found. Make sure both exist.")

df = pd.read_csv(CSV_PATH)
embeddings = np.load(EMB_PATH)

# ---- Build FAISS index ----
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# ---- Load CLIP model ----
model = SentenceTransformer("clip-ViT-B-32")

# ---- Search endpoint ----
@app.post("/search")
async def search_image(file: UploadFile = File(...), top_k: int = 10):
    try:
        # Read uploaded image
        image_bytes = await file.read()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")

    # Encode image
    query_emb = model.encode(img)
    query_emb = np.expand_dims(query_emb, axis=0)

    # Search FAISS
    distances, indices = index.search(query_emb, top_k)

    # Build results
    results = []
    for idx, score in zip(indices[0], distances[0]):
        row = df.iloc[idx]
        results.append({
            "id": int(row["id"]),
            "name": row["product_name"],
            "price": row["product_price"],
            "main_photo_url": IMAGE_URL_BASE + row["main_photo"],
            "product_url": f"https://ladonna.com.bd/product/product_description?id={row['id']}",
            "score": float(score)
        })

    return {"results": results}

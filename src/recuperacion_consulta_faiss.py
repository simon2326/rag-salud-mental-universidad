# -*- coding: utf-8 -*-
"""
Created on Wed Sep 10 15:33:01 2025

@author: 000010478
"""

# query_search.py
import os
import numpy as np
import pandas as pd
import faiss
from openai import AzureOpenAI

# ========= Config =========
FAISS_INDEX_PATH = "./faiss_index.faiss"
PARQUET_PATH     = "./chunks.parquet"

# ========= Cliente Azure OpenAI =========
endpoint = "https://pnl-maestria.openai.azure.com/"
deployment = "text-embedding-3-small"
client = AzureOpenAI(
    api_key="AZURE_OPENAI_API_KEY_PLACEHOLDER",
    api_version="2024-10-21",   # o 2025-04-01-preview si quieres preview
    azure_endpoint=endpoint
)

def get_embedding(text: str) -> np.ndarray:
    """Obtiene el embedding de una sola frase."""
    resp = client.embeddings.create(
        input=[text],
        model=deployment
    )
    emb = np.array(resp.data[0].embedding, dtype="float32")
    # FAISS espera (n, d); convertimos a forma 2D
    emb = np.ascontiguousarray(emb.reshape(1, -1))
    # Normalizamos para usar coseno con IndexFlatL2 (distancia L2 entre unit vectors)
    faiss.normalize_L2(emb)
    return emb

def load_artifacts():
    """Carga índice FAISS y el dataframe de chunks (parquet)."""
    index = faiss.read_index(FAISS_INDEX_PATH)
    df = pd.read_parquet(PARQUET_PATH)
    df = df.rename(columns={df.columns[0]: "text"})
    return index, df

def cosine_from_l2_dist(dist_sq: np.ndarray) -> np.ndarray:
    """
    Convierte distancia L2^2 entre vectores unitarios a similitud coseno.
    Para vectores normalizados:  ||a-b||^2 = 2(1 - cos_sim)  =>  cos_sim = 1 - dist^2/2
    """
    return 1.0 - dist_sq / 2.0

def search(query: str, k: int = 10):
    """Busca los k más cercanos al embedding de la consulta y devuelve un DataFrame con resultados."""
    index, df = load_artifacts()
    q = get_embedding(query)
    # IndexFlatIP hace producto interno
    D, I = index.search(q, k)  
    D = D[0]
    I = I[0]
    #print(D, I)
    
    # Extrae los chunks
    results = df.iloc[I].copy()
    # Calcula similitud coseno a partir de L2^2 (como normalizaste embeddings)
    results["cosine_sim"] = D
    results["faiss_id"] = I
    # Ordena por mayor similitud
    results = results.sort_values("cosine_sim", ascending=False).reset_index(drop=True)
    return results


pregunta = "¿Cuál es la manera como un estudiante puede quedar por fuera de un programa?"
topk = 10
res = search(pregunta, k=topk)


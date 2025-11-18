# -*- coding: utf-8 -*-
"""
Funciones de recuperación: permiten cargar artefactos FAISS/chunks por modelo
y ejecutar búsqueda estándar o con MMR.

Autor original: Roberto Hincapié
Adaptado por: Simón Correa Marín
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Tuple

import faiss
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

DATA_DIR = Path("data")
INDICES_DIR = Path("indices")
DEFAULT_MODEL = "base"
DEFAULT_MMR_MODEL = "overlap"
DEFAULT_K = 5
EMBEDDING_DEPLOYMENT = os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")

client = AzureOpenAI(
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
)

_ARTIFACT_CACHE: Dict[str, Tuple[faiss.Index, pd.DataFrame]] = {}


def artifacts_paths(model_name: str) -> Tuple[Path, Path]:
    return (
        INDICES_DIR / f"faiss_index_{model_name}.faiss",
        DATA_DIR / f"chunks_{model_name}.parquet",
    )


def load_artifacts(model_name: str = DEFAULT_MODEL) -> Tuple[faiss.Index, pd.DataFrame]:
    """Carga y cachea el índice FAISS junto al DataFrame de chunks."""
    if model_name in _ARTIFACT_CACHE:
        return _ARTIFACT_CACHE[model_name]

    index_path, data_path = artifacts_paths(model_name)
    if not index_path.exists() or not data_path.exists():
        raise FileNotFoundError(f"No se encontraron artefactos para '{model_name}'")

    index = faiss.read_index(str(index_path))
    df = pd.read_parquet(data_path)
    if "text" not in df.columns:
        first_col = df.columns[0]
        df = df.rename(columns={first_col: "text"})

    _ARTIFACT_CACHE[model_name] = (index, df)
    return index, df


def get_embedding(text: str) -> np.ndarray:
    """Obtiene y normaliza el embedding de la consulta."""
    resp = client.embeddings.create(input=[text], model=EMBEDDING_DEPLOYMENT)
    emb = np.array(resp.data[0].embedding, dtype="float32")
    emb = np.ascontiguousarray(emb.reshape(1, -1))
    faiss.normalize_L2(emb)
    return emb


def _build_results(df: pd.DataFrame, indices: np.ndarray, scores: np.ndarray) -> pd.DataFrame:
    results = df.iloc[indices].copy()
    results = results.reset_index(drop=True)
    results["cosine_sim"] = scores
    results["faiss_id"] = indices
    return results


def search(query: str, model_name: str = DEFAULT_MODEL, k: int = DEFAULT_K) -> pd.DataFrame:
    """
    Búsqueda directa por similitud para los modelos 0, 1 y 2.
    """
    index, df = load_artifacts(model_name)
    query_vec = get_embedding(query)
    scores, idxs = index.search(query_vec, k)
    scores = scores[0]
    idxs = idxs[0]
    return _build_results(df, idxs, scores).sort_values("cosine_sim", ascending=False).reset_index(drop=True)


def search_mmr(
    query: str,
    model_name: str = DEFAULT_MMR_MODEL,
    k: int = DEFAULT_K,
    fetch_k: int = 20,
    lambda_mult: float = 0.5,
) -> pd.DataFrame:
    """
    Recuperación con Maximum Marginal Relevance para reducir redundancia.
    - fetch_k controla cuántos candidatos iniciales se toman del índice.
    - lambda_mult balancea relevancia vs. diversidad (0.5 = equilibrio).
    """
    if fetch_k < k:
        raise ValueError("fetch_k debe ser >= k")

    index, df = load_artifacts(model_name)
    query_vec = get_embedding(query)
    scores, idxs = index.search(query_vec, fetch_k)
    candidate_scores = scores[0]
    candidate_ids = idxs[0]

    # Recuperamos los embeddings normalizados de cada candidato desde el índice
    candidate_vectors = np.array([index.reconstruct(int(idx)) for idx in candidate_ids])
    query_vec_flat = query_vec[0]

    selected: list[int] = []
    remaining = list(range(len(candidate_ids)))

    while remaining and len(selected) < k:
        best_idx = None
        best_score = -np.inf
        for candidate_idx in remaining:
            candidate_vec = candidate_vectors[candidate_idx]
            relevance = float(np.dot(candidate_vec, query_vec_flat))
            if not selected:
                diversity = 0.0
            else:
                diversity = max(
                    float(np.dot(candidate_vec, candidate_vectors[sel_idx])) for sel_idx in selected
                )
            mmr_score = lambda_mult * relevance - (1 - lambda_mult) * diversity
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = candidate_idx
        selected.append(best_idx)
        remaining.remove(best_idx)

    selected_ids = candidate_ids[selected]
    selected_scores = candidate_scores[selected]
    results = _build_results(df, selected_ids, selected_scores)
    results["mmr_rank"] = range(1, len(results) + 1)
    return results


if __name__ == "__main__":
    pregunta = "¿Cuáles son los principales factores de riesgo para la salud mental universitaria?"
    print("=== Búsqueda estándar (base) ===")
    print(search(pregunta).head())
    print("\n=== Búsqueda MMR (overlap) ===")
    print(search_mmr(pregunta, k=5, lambda_mult=0.6))


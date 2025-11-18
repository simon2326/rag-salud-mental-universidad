# -*- coding: utf-8 -*-
"""
Script principal de indexación: genera embeddings con Azure OpenAI para
distintas configuraciones de chunking y guarda los artefactos FAISS/parquet.

Autor original: Roberto Hincapié
Adaptado por: Simón Correa Marín
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, List

import faiss  # pip install faiss-cpu
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import AzureOpenAI

from clases_textos import chunk_text, collect_files, extract_text_by_ext

load_dotenv()

DOCS_DIR = Path("documentos")
DATA_DIR = Path("data")
INDICES_DIR = Path("indices")
EMBEDDING_DEPLOYMENT = os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")
CONFIGS: Dict[str, Dict[str, int]] = {
    "base": {"chunk_size": 300, "overlap": 50},
    "small": {"chunk_size": 200, "overlap": 40},
    "overlap": {"chunk_size": 300, "overlap": 120},
}
BATCH_SIZE = 32


def build_client() -> AzureOpenAI:
    return AzureOpenAI(
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    )


def batched(iterable: List[str], batch_size: int) -> Iterable[List[str]]:
    for start in range(0, len(iterable), batch_size):
        yield iterable[start : start + batch_size]


def embed_chunks(client: AzureOpenAI, chunks: List[str]) -> List[List[float]]:
    vectors: List[List[float]] = []
    for batch in batched(chunks, BATCH_SIZE):
        response = client.embeddings.create(input=batch, model=EMBEDDING_DEPLOYMENT)
        vectors.extend([item.embedding for item in response.data])
    return vectors


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    INDICES_DIR.mkdir(parents=True, exist_ok=True)


def process_config(client: AzureOpenAI, name: str, config: Dict[str, int]) -> None:
    chunk_records: List[Dict[str, str]] = []
    embeddings: List[List[float]] = []

    print(f"\nProcesando configuración '{name}' ({config['chunk_size']} tokens, overlap {config['overlap']})")
    files = collect_files(DOCS_DIR)
    if not files:
        raise FileNotFoundError(f"No se encontraron documentos en {DOCS_DIR.resolve()}")

    for doc_path in files:
        text = extract_text_by_ext(doc_path)
        if not text:
            print(f"  - {doc_path.name}: sin texto legible, se omite.")
            continue

        chunks = chunk_text(text, config["chunk_size"], config["overlap"])
        if not chunks:
            print(f"  - {doc_path.name}: no se generaron chunks, se omite.")
            continue

        print(f"  - {doc_path.name}: {len(chunks)} chunks")
        doc_embeddings = embed_chunks(client, chunks)
        embeddings.extend(doc_embeddings)

        for idx, chunk in enumerate(chunks):
            chunk_records.append(
                {
                    "doc_id": doc_path.stem,
                    "chunk_index": idx,
                    "text": chunk,
                    "source_path": str(doc_path),
                }
            )

    if not embeddings:
        raise RuntimeError(f"No se generaron embeddings para la configuración '{name}'")

    embeddings_arr = np.ascontiguousarray(np.array(embeddings).astype("float32"))
    faiss.normalize_L2(embeddings_arr)
    index = faiss.IndexFlatIP(embeddings_arr.shape[1])
    index.add(embeddings_arr)

    df = pd.DataFrame(chunk_records)
    data_path = DATA_DIR / f"chunks_{name}.parquet"
    index_path = INDICES_DIR / f"faiss_index_{name}.faiss"

    df.to_parquet(data_path)
    faiss.write_index(index, str(index_path))
    print(f"  -> Guardado {len(chunk_records)} chunks en {data_path}")
    print(f"  -> Índice FAISS almacenado en {index_path}")


def main() -> None:
    ensure_dirs()
    client = build_client()
    for name, cfg in CONFIGS.items():
        process_config(client, name, cfg)


if __name__ == "__main__":
    main()


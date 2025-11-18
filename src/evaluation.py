# -*- coding: utf-8 -*-
"""
Evaluación de los modelos de recuperación: calcula recall@5 y tamaño medio de
contexto para los modelos base, small, overlap y mmr usando el ground truth.

Autor original: Roberto Hincapié
Adaptado por: Simón Correa Marín
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Iterable, List

import numpy as np
import pandas as pd
import tiktoken

if __package__:
    from .recuperacion_consulta_faiss import DEFAULT_K, search, search_mmr
else:
    from recuperacion_consulta_faiss import DEFAULT_K, search, search_mmr


QUERIES: List[Dict[str, str]] = [
    # --- 02.pdf: Revisión de literatura en universitarios colombianos ---
    {
        "id": 1,
        "question": "¿Por qué la salud mental de los estudiantes universitarios se considera un tema de creciente interés social en Colombia?",
        "doc_id": "02.pdf",
    },
    {
        "id": 2,
        "question": "¿Qué impacto tuvo la pandemia de COVID-19 en la salud mental de los estudiantes universitarios según la revisión de literatura colombiana 2018–2023?",
        "doc_id": "02.pdf",
    },
    {
        "id": 3,
        "question": "¿Cuál fue el objetivo principal de la revisión de la literatura sobre salud mental en universitarios colombianos entre 2018 y 2023?",
        "doc_id": "02.pdf",
    },
    {
        "id": 4,
        "question": "¿Qué tipo de enfoques metodológicos predominan en los estudios colombianos sobre salud mental en estudiantes universitarios?",
        "doc_id": "02.pdf",
    },
    {
        "id": 5,
        "question": "¿Qué se concluye sobre la diversidad de instrumentos psicométricos utilizados para evaluar salud mental en universitarios colombianos?",
        "doc_id": "02.pdf",
    },
    {
        "id": 6,
        "question": "¿Qué recomendación hace la revisión respecto a la necesidad de estudios cualitativos sobre salud mental en estudiantes universitarios?",
        "doc_id": "02.pdf",
    },
    # --- 05.pdf: Revisión sistemática ---
    {
        "id": 7,
        "question": "¿Cómo define la Organización Mundial de la Salud el concepto general de salud citado en la revisión sistemática de universitarios de América Latina?",
        "doc_id": "05.pdf",
    },
    {
        "id": 8,
        "question": "¿Qué entiende la revisión sistemática por salud mental desde una perspectiva biopsicosocial?",
        "doc_id": "05.pdf",
    },
    {
        "id": 9,
        "question": "¿Cuáles son los seis factores de la escala de Salud Mental Positiva propuesta por Lluch que se mencionan en la revisión latinoamericana?",
        "doc_id": "05.pdf",
    },
    {
        "id": 10,
        "question": "¿Qué características metodológicas generales se describen para el estudio de salud mental positiva en estudiantes del área de la salud de Cartagena dentro de la revisión sistemática?",
        "doc_id": "05.pdf",
    },
    {
        "id": 11,
        "question": "¿Qué instrumentos se mencionan en la revisión para evaluar calidad de vida, ansiedad y depresión en estudiantes universitarios usuarios de servicios de salud mental?",
        "doc_id": "05.pdf",
    },
    {
        "id": 12,
        "question": "¿Qué factores de riesgo para la mala salud mental en universitarios se identifican en el estudio sobre usuarios de servicios de salud mental reseñado en la revisión sistemática?",
        "doc_id": "05.pdf",
    },
    {
        "id": 13,
        "question": "Según la revisión sistemática, ¿por qué se considera necesario seguir investigando la salud mental de los universitarios en América Latina?",
        "doc_id": "05.pdf",
    },
    # --- 07.pdf: Documento REUPS ---
    {
        "id": 14,
        "question": "¿Qué papel se propone para los servicios de atención psicológica universitarios en la promoción del bienestar emocional según las recomendaciones de la REUPS?",
        "doc_id": "07.pdf",
    },
    {
        "id": 15,
        "question": "¿Qué tipo de espacios físicos se recomienda diseñar en las universidades para favorecer la relajación y las pausas saludables del estudiantado?",
        "doc_id": "07.pdf",
    },
    {
        "id": 16,
        "question": "¿Qué se sugiere en el documento respecto a la formación del profesorado y del personal de administración y servicios (PDI y PAS) en competencias socioemocionales?",
        "doc_id": "07.pdf",
    },
    {
        "id": 17,
        "question": "Según el informe, ¿cómo ha influido la pandemia de COVID-19 en el malestar psicológico y en la necesidad de atención a la salud mental en el ámbito universitario?",
        "doc_id": "07.pdf",
    },
    {
        "id": 18,
        "question": "¿Qué relación describe la REUPS entre el uso del teléfono móvil, el estilo de vida y el bienestar psicológico del estudiantado universitario durante la pandemia?",
        "doc_id": "07.pdf",
    },
    # --- 10.pdf: Cartagena ---
    {
        "id": 19,
        "question": "¿Cuál es la pregunta problema planteada en el estudio sobre salud mental positiva en estudiantes del área de la salud de una universidad de Cartagena?",
        "doc_id": "10.pdf",
    },
    {
        "id": 20,
        "question": "¿Cuál es el objetivo general del trabajo sobre salud mental positiva en estudiantes del área de la salud en Cartagena?",
        "doc_id": "10.pdf",
    },
    {
        "id": 21,
        "question": "¿Qué tipo de estudio y diseño metodológico se utiliza en la investigación de salud mental positiva en estudiantes del área de la salud de Cartagena?",
        "doc_id": "10.pdf",
    },
    {
        "id": 22,
        "question": "¿Cuáles son los principales instrumentos de recolección de información empleados en el estudio de salud mental positiva de la Universidad de Cartagena?",
        "doc_id": "10.pdf",
    },
    {
        "id": 23,
        "question": "¿Qué concluye el estudio de Cartagena sobre los niveles de salud mental positiva y sus factores en los estudiantes evaluados?",
        "doc_id": "10.pdf",
    },
]


MODELS = [
    {"label": "base", "type": "search", "kwargs": {"model_name": "base", "k": DEFAULT_K}},
    {"label": "small", "type": "search", "kwargs": {"model_name": "small", "k": DEFAULT_K}},
    {"label": "overlap", "type": "search", "kwargs": {"model_name": "overlap", "k": DEFAULT_K}},
    {
        "label": "mmr",
        "type": "mmr",
        "kwargs": {"model_name": "overlap", "k": DEFAULT_K, "fetch_k": 20, "lambda_mult": 0.6},
    },
]

ENCODING = tiktoken.encoding_for_model("text-embedding-3-small")


def normalize_doc_id(value: str) -> str:
    name = Path(str(value)).name
    if not name.lower().endswith(".pdf"):
        name = f"{name}.pdf"
    return name.lower()


def compute_context_len(texts: Iterable[str]) -> int:
    if isinstance(texts, pd.Series):
        iterable = texts.tolist()
    else:
        iterable = list(texts)
    return int(sum(len(ENCODING.encode(text or "")) for text in iterable))


def evaluate_model(label: str, retrieval_fn: Callable[[str], pd.DataFrame]) -> Dict[str, float]:
    hits = 0
    context_lengths: List[int] = []

    for query in QUERIES:
        df = retrieval_fn(query["question"])
        df = df.head(DEFAULT_K)
        if "doc_id" in df.columns:
            docs = df["doc_id"].fillna("").map(normalize_doc_id)
        elif "source_path" in df.columns:
            docs = df["source_path"].fillna("").map(normalize_doc_id)
        else:
            docs = pd.Series(dtype=str)
        target = normalize_doc_id(query["doc_id"])
        if target in docs.values:
            hits += 1
        texts = df["text"] if "text" in df.columns else pd.Series(dtype=str)
        context_lengths.append(compute_context_len(texts))

    recall = hits / len(QUERIES)
    avg_context = float(np.mean(context_lengths)) if context_lengths else 0.0
    return {"modelo": label, "recall@5": recall, "avg_context_len": avg_context}


def make_retrieval_fn(spec: Dict) -> Callable[[str], pd.DataFrame]:
    if spec["type"] == "mmr":
        return lambda question: search_mmr(question, **spec["kwargs"])
    return lambda question: search(question, **spec["kwargs"])


def run_experiments() -> pd.DataFrame:
    rows = []
    for spec in MODELS:
        print(f"Evaluando modelo '{spec['label']}'")
        fn = make_retrieval_fn(spec)
        rows.append(evaluate_model(spec["label"], fn))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    results = run_experiments()
    print("\nResultados globales:")
    print(results)

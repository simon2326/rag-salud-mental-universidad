# -*- coding: utf-8 -*-
"""
Utilidades para ingestión y chunking de documentos del proyecto.
Se encargan de leer PDFs/TXT, normalizar su contenido y generar
chunks configurables que luego serán indexados dentro del pipeline RAG.

Autor original: Roberto Hincapié
Adaptado por: Simón Correa Marín
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence
import re

import tiktoken
from pypdf import PdfReader  # pip install pypdf
from pathlib import Path

# Carpeta por defecto donde almacenamos los 16 documentos del tema
DEFAULT_DOCUMENT_DIR = Path("documentos")
# Extensiones soportadas
SUPPORTED_EXTENSIONS: Sequence[str] = (".pdf", ".txt", ".md")


@dataclass
class ChunkConfig:
    """Configura el esquema de partición de los documentos."""

    chunk_size: int = 300
    overlap: int = 50
    model_name: str = "text-embedding-3-small"

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size debe ser positivo")
        if self.overlap < 0:
            raise ValueError("overlap no puede ser negativo")
        if self.overlap >= self.chunk_size:
            raise ValueError("overlap debe ser menor que chunk_size")


def normalize_text(text: str) -> str:
    """Limpia saltos de línea y espacios para homogeneizar los chunks."""
    cleaned = text.replace("\x00", " ")
    cleaned = re.sub(r"\r\n?", "\n", cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()

def read_txt(path: Path) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return normalize_text(f.read())


def read_pdf(path: Path) -> str:
    reader = PdfReader(path)
    texts: List[str] = []
    for page in reader.pages:
        try:
            texts.append(page.extract_text() or "")
        except Exception:
            # Algunos PDFs pueden tener páginas sin texto legible
            texts.append("")
    return normalize_text("\n".join(texts))


def _chunk_spans(length: int, chunk_size: int, overlap: int) -> List[tuple[int, int]]:
    """Devuelve los intervalos de tokens que conforman cada chunk."""
    spans: List[tuple[int, int]] = []
    step = max(1, chunk_size - overlap)
    for start in range(0, length, step):
        end = min(start + chunk_size, length)
        spans.append((start, end))
        if end >= length:
            break
    return spans


def chunk_text(
    text: str,
    chunk_size: int,
    overlap: int,
    model_name: str = "text-embedding-3-small",
) -> List[str]:
    """
    Retorna únicamente el texto de cada chunk.
    """
    encoding = tiktoken.encoding_for_model(model_name)
    tokens = encoding.encode(text)
    spans = _chunk_spans(len(tokens), chunk_size, overlap)
    return [encoding.decode(tokens[start:end]) for start, end in spans]


def chunk_document(
    text: str,
    doc_id: str,
    config: ChunkConfig,
    source_path: Optional[Path] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Versión extendida que adjunta metadatos relevantes de cada chunk para el RAG.
    """
    encoding = tiktoken.encoding_for_model(config.model_name)
    tokens = encoding.encode(text)
    spans = _chunk_spans(len(tokens), config.chunk_size, config.overlap)
    metadata = extra_metadata or {}
    records: List[Dict[str, Any]] = []
    for idx, (start, end) in enumerate(spans):
        chunk_tokens = tokens[start:end]
        chunk_text_val = encoding.decode(chunk_tokens)
        record = {
            "chunk_id": f"{doc_id}_{idx}",
            "text": chunk_text_val,
            "doc_id": doc_id,
            "chunk_index": idx,
            "start_token": start,
            "end_token": end,
            "token_count": end - start,
            "source_path": str(source_path) if source_path else None,
            "metadata": metadata,
        }
        records.append(record)
    return records


def collect_files(
    input_dir: Path | str = DEFAULT_DOCUMENT_DIR,
    extensions: Iterable[str] = SUPPORTED_EXTENSIONS,
) -> List[Path]:
    """
    Recorre la carpeta de documentos y devuelve las rutas ordenadas.
    """
    base_path = Path(input_dir)
    if not base_path.exists():
        return []
    files: List[Path] = []
    for ext in extensions:
        files.extend(base_path.rglob(f"*{ext}"))
    return sorted(p for p in files if p.is_file())


def extract_text_by_ext(path: Path | str) -> str:
    path_obj = Path(path)
    ext = path_obj.suffix.lower()
    if ext == ".pdf":
        return read_pdf(path_obj)
    if ext in [".txt", ".md"]:
        return read_txt(path_obj)
    return ""

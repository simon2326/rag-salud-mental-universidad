# -*- coding: utf-8 -*-
"""
Created on Mon Sep  8 21:58:25 2025

@author: 000010478
"""
from typing import List, Dict, Iterable, Tuple
import tiktoken
from pypdf import PdfReader #pip install pypdf
import glob
from pathlib import Path

def read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def read_pdf(path: str) -> str:
    reader = PdfReader(path)
    texts = []
    for page in reader.pages:
        try:
            texts.append(page.extract_text() or "")
        except Exception:
            texts.append("")
    return "\n".join(texts)



def chunk_text(text: str, chunk_size: int, overlap: int, 
               model_name: str = "text-embedding-3-small") -> List[str]:
    """
    Returns:
        List[str]: lista de chunks de texto
    """
    encoding = tiktoken.encoding_for_model(model_name)
    tokens = encoding.encode(text)

    chunks = []
    step = max(1, chunk_size - overlap)

    for start in range(0, len(tokens), step):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)
        if end >= len(tokens):
            break

    return chunks


def collect_files(input_dir: str) -> List[str]:
    exts = ["**/*.pdf", "**/*.txt"]
    files = []
    for pat in exts:
        files.extend(glob.glob(str(Path(input_dir) / pat), recursive=True))
    # quitar duplicados y ordenar
    files = sorted(list(set(files)))
    files=['./documentos/'+fi.split('\\')[1] for fi in files]
    return files

def extract_text_by_ext(path: str) -> str:
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        return read_pdf(path)
    elif ext in [".txt", ".md"]:
        return read_txt(path)
    else:
        return ""
    
    
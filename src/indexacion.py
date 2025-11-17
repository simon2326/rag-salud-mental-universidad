# -*- coding: utf-8 -*-
"""
Created on Mon Sep  8 21:34:09 2025

@author: 000010478
"""

import os
import io
import glob
import math
import uuid
import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Iterable, Tuple
import pandas as pd
from tqdm import tqdm
from clases_textos import read_txt, read_pdf, chunk_text, collect_files, extract_text_by_ext
import numpy as np
from openai import AzureOpenAI
import faiss  # pip install faiss-cpu


endpoint = "https://pnl-maestria.openai.azure.com/"
deployment = "text-embedding-3-small"

client = AzureOpenAI(
    api_key="AZURE_OPENAI_API_KEY_PLACEHOLDER",
    api_version="2024-10-21",   # o 2025-04-01-preview si quieres preview
    azure_endpoint=endpoint
)

# --- Lectura de archivos ---
chunks=[]
chunk_size=300
overlap=50
embeddings=[]
folder='./documentos/'

for file in collect_files(folder):
    ch=chunk_text(extract_text_by_ext(file), chunk_size, overlap)
    print('Documento: ',file, ', compuesto de ',len(ch), ' chunks')
    chunks=chunks+ch
    response = client.embeddings.create(
        input=ch,
        model=deployment   # ¡OJO! en Azure siempre es el nombre del deployment
    )
    embs=[d.embedding for d in response.data]
    embeddings=embeddings+embs

#Creación de un dataframe de los chunks
data=pd.DataFrame(data=chunks)

#Dimensión de los embeddings para poder crear el índice
d=len(embeddings[0])        
#Conversión de los embeddings a arreglos continuos en memoria y tipo float32
embeddings=np.ascontiguousarray(np.array(embeddings).astype('float32'))
#Creación del índice, FlatL2 es orientado a distancias basadas en similitud del coseno
index = faiss.IndexFlatIP(d)
#Normalización de los índices para poder calcular la similitud del coseno
faiss.normalize_L2(embeddings)
#Adicionamos los chunks a los embeddings
index.add(embeddings)

# Guardar artefactos
out_index = "./faiss_index.faiss"
faiss.write_index(index, str(out_index))
data.to_parquet('chunks.parquet') #Guardado del dataframe de chunks


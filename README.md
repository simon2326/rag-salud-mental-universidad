# rag-salud-mental-universidad

Proyecto final de RAG desarrollado por **Simón Correa Marín**. Implementa un pipeline completo para responder preguntas sobre salud mental y bienestar en estudiantes universitarios usando FAISS y Azure OpenAI. Se comparan configuraciones de chunking (base, small, overlap) y un modelo con reducción de redundancia (MMR), evaluando `recall@5` y el tamaño medio del contexto.

## Requisitos

- Python 3.10+
- Dependencias listadas en `requirements.txt`
- Variables de entorno en `.env` (`AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_API_VERSION`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_EMBEDDING_DEPLOYMENT`, `AZURE_OPENAI_CHAT_DEPLOYMENT`)

Instalación rápida:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # y completa tus credenciales de Azure OpenAI
```

## Uso vía notebook (laboratorio central)

1. Abre `notebooks/experimentacion.ipynb`.
2. Ejecuta las celdas en orden:
   - **Configuración**: carga `.env`, detecta la raíz del repo y ajusta `sys.path`.
   - **Revisión de documentos**: lista y chunkéa un PDF para validar la ingestión.
   - **Indexación**: corre `python src/indexacion.py` para generar embeddings e índices FAISS de las configuraciones `base`, `small` y `overlap`.
   - **Comparación de recuperaciones**: ejecuta `search` y `search_mmr` para ver los `top-k`.
   - **Evaluación**: ejecuta `run_experiments()` (usa los ground truth definidos en `src/evaluation.py`).
   - **Demo RAG**: usa `generate_answer()` para mostrar pregunta → chunks → respuesta del LLM; ideal para capturas del informe.

> Todas las rutas relativas funcionan desde el notebook gracias a la celda de configuración (se cambia temporalmente el `cwd` a la raíz del proyecto).

## Uso desde scripts

- `python src/indexacion.py`: genera `data/chunks_<modelo>.parquet` e índices FAISS.
- `python src/evaluation.py`: imprime la tabla de métricas (baseline vs. variantes).
- `python src/consulta_rag.py`: demo en línea de comandos para una pregunta.

## Resultados clave

| modelo  | recall@5 | avg_context_len |
|---------|----------|-----------------|
| base    | 0.91     | 1500            |
| small   | 0.91     | 1000            |
| overlap | 0.87     | 1500            |
| mmr     | **0.96** | 1484            |

El enfoque MMR (selección diversificada sobre el índice `overlap`) ofreció el mejor equilibrio entre precisión y tamaño de contexto. `small` empató en recall con `base` pero con muchos menos tokens, por lo que es una buena alternativa cuando se desea optimizar costos.

## Conclusiones

- El pipeline es reproducible end-to-end y se centraliza en el notebook para facilitar experimentación y documentación.
- Las configuraciones de chunking afectan directamente la cobertura y el tamaño del contexto; los experimentos demuestran que no siempre “más overlap” implica mejor recall.
- Incorporar MMR reduce la redundancia y eleva el recall sin incrementar el contexto medio, por lo que se recomienda como configuración final para el informe y las demos.
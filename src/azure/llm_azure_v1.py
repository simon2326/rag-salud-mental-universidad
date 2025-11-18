# -*- coding: utf-8 -*-
"""
Script rápido para verificar la conexión con Azure OpenAI usando variables
de entorno definidas en .env. Ejecutar antes de los experimentos con FAISS.

Autor original: Roberto Hincapié
Adaptado por: Simón Correa Marín
"""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv
from openai import AzureOpenAI


def build_client() -> AzureOpenAI:
    load_dotenv()
    required = ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_API_VERSION", "AZURE_OPENAI_ENDPOINT"]
    missing = [var for var in required if not os.environ.get(var)]
    if missing:
        raise EnvironmentError(f"Faltan variables en .env: {', '.join(missing)}")

    return AzureOpenAI(
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    )


def main() -> None:
    client = build_client()
    response = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[
            {
                "role": "system",
                "content": (
                    "Eres un asistente experto en RAG aplicado a salud mental universitaria. "
                    "Responde en menos de 120 tokens."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Explica brevemente por qué validar la conexión con Azure OpenAI "
                    "es importante antes de indexar documentos."
                ),
            },
        ],
        max_tokens=200,
    )
    print(response.choices[0].message.content)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error al probar Azure OpenAI: {exc}", file=sys.stderr)
        sys.exit(1)

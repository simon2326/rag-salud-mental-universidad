# -*- coding: utf-8 -*-
"""
Created on Tue Oct  7 17:42:32 2025

@author: 000010478
"""

from openai import AzureOpenAI

client = AzureOpenAI(
    api_key="AZURE_OPENAI_API_KEY_PLACEHOLDER",
    api_version='2024-12-01-preview',
    azure_endpoint='https://pnl-maestria.openai.azure.com/'
    )


response = client.chat.completions.create(
    model="gpt-4.1-nano",# "o4-mini"
    messages=[
        {"role": "system",
         "content": "Considera que eres un experto en áreas de analítica y ciencia de datos y das respuestas amplias pero al mismo tiempo concretas. Da una respuesta de máximo 200 tokens"},
        {"role": "user","content": "Explica de manera resumida, considerando 3 párrafos: qué es un agente."}
        ]
)

print(response.choices[0].message.content)
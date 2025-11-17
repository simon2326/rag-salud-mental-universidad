import numpy as np
from openai import AzureOpenAI

from recuperacion_consulta_faiss import search

client = AzureOpenAI(
    api_key="AZURE_OPENAI_API_KEY_PLACEHOLDER",
    api_version='2024-12-01-preview',
    azure_endpoint='https://pnl-maestria.openai.azure.com/'
    )

query = 'Cómo puede un estudiante perder su calidad de estudiante?'
res=search(query)
results = list(res['text'].values)

context = "\n\n".join(results)
prompt = f"Contexto:\n{context}\n\nPregunta: {query}\nRespuesta:"

# Nueva forma de hacer la petición
response = client.chat.completions.create(
    model="gpt-4.1-nano",# "o4-mini"
    messages=[
        {"role": "system", "content": "Eres un experto en educación e investigación en ingeniería."},
        {"role": "user", "content": prompt}
    ]
)

print(response.choices[0].message.content)




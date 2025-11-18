import os
from typing import Literal

from dotenv import load_dotenv
from openai import AzureOpenAI

if __package__:
    from .recuperacion_consulta_faiss import DEFAULT_K, search, search_mmr
else:
    from recuperacion_consulta_faiss import DEFAULT_K, search, search_mmr

load_dotenv()

CHAT_DEPLOYMENT = os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4.1-nano")

client = AzureOpenAI(
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
)


def retrieve_chunks(
    question: str,
    model: Literal["base", "small", "overlap", "mmr"] = "base",
    k: int = DEFAULT_K,
    show_chunks: bool = True,
):
    if model == "mmr":
        df = search_mmr(question, model_name="overlap", k=k, fetch_k=20, lambda_mult=0.6)
    else:
        df = search(question, model_name=model, k=k)

    if show_chunks:
        print("\n--- Chunks seleccionados ---")
        for idx, (_, row) in enumerate(df.head(k).iterrows(), start=1):
            print(f"[{idx}] doc={row.get('doc_id')} | cos_sim={row.get('cosine_sim'):.3f}")
            snippet = (row.get("text") or "")[:400].strip()
            print(snippet, "\n")
    return df.head(k)


def build_prompt(question: str, context_chunks) -> str:
    context = "\n\n".join(
        f"Documento: {row.get('doc_id')} (similaridad {row.get('cosine_sim'):.3f})\n{row.get('text') or ''}"
        for _, row in context_chunks.iterrows()
    )
    return (
        "Eres un experto en salud mental universitaria. Responde citando únicamente el contexto proporcionado. "
        "Si la respuesta no está en el contexto, indica que falta información.\n\n"
        f"Contexto:\n{context}\n\nPregunta: {question}\nRespuesta:"
    )


def generate_answer(question: str, model: str = "base", k: int = DEFAULT_K, show_chunks: bool = True) -> str:
    chunks = retrieve_chunks(question, model=model, k=k, show_chunks=show_chunks)
    prompt = build_prompt(question, chunks)
    print("\n--- Pregunta ---")
    print(question)

    response = client.chat.completions.create(
        model=CHAT_DEPLOYMENT,
        messages=[
            {"role": "system", "content": "Eres un asistente experto en RAG aplicado a salud mental universitaria."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=350,
    )

    answer = response.choices[0].message.content
    print("\n--- Respuesta ---")
    print(answer)
    return answer


if __name__ == "__main__":
    sample_question = "¿cuales son las principales causas de ansiedad en estudiantes universitarios?"
    generate_answer(sample_question, model="mmr", k=5, show_chunks=True)

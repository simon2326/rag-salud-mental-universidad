# main_chat_sync.py
# -*- coding: utf-8 -*-
import os
from openai import AzureOpenAI

client = AzureOpenAI(
    api_key="AZURE_OPENAI_API_KEY_PLACEHOLDER",
    api_version='2024-12-01-preview',
    azure_endpoint='https://pnl-maestria.openai.azure.com/'
    )

conversacion = []  # [{"role": "user"|"assistant", "content": str}]
uso=[]

def ask(model: str, user_text: str, instructions: str | None = None) -> str:
    response = client.chat.completions.create(
        model="gpt-4.1-nano",# "o4-mini"
        messages=[{"role": "system", "content": instructions}]+conversacion
    )
    uso.append({'prompt':response.usage.prompt_tokens, 'respuesta':response.usage.completion_tokens, 'total':response.usage.total_tokens})
    return response.choices[0].message.content

def main():
    model = "o4-mini", #"gpt-4o-nano"
    instrucciones = "Responde en español, considerando la información más completa que tengas, pero sin alucinar"
    print("Chat con GPT (escribe 'salir' para terminar)\n" + "-"*50)
    while True:
        usuario = input("\nTú: ").strip()
        if usuario.lower() in {"salir", "exit", "quit"}:
            print("¡Hasta luego!")
            break

        conversacion.append({"role": "user", "content": usuario})
        try:
            respuesta = ask(model, usuario, instrucciones)
            print(f"\nGPT:\n{respuesta}")
            conversacion.append({"role": "assistant", "content": respuesta})
        except Exception as e:
            print(f"\nError: {e}")

if __name__ == "__main__":
    main()
print(conversacion)
print(uso)
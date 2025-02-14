# embedding.py
from openai import OpenAI   
from os import environ as env

api_key = env.get("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)

def embedding(input: str):
    return client.embeddings.create(
        model="text-embedding-ada-002",
        input=input,
    ).data[0].embedding
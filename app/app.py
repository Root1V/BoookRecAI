from openai import OpenAI
from dotenv import load_dotenv
from app.config import Config
import pandas as pd
import numpy as np
import os
import tiktoken


class App:

    def __init__(self) -> None:
        self.config = Config.get_all()
        self.client = OpenAI()

    def load_data(self):
        df = pd.read_csv("data/book.csv")
        df = df.dropna(subset=["Descripción"]).reset_index(drop=True)
        # df = df.sample(n=20, random_state=1)
        print(df)
        self.get_coste_embeddings(df)
        return df

    def one_run(self):
        df = self.load_data()
        self.save_embeddings(df, "embeddings.csv")

    def get_coste_embeddings(self, df):
        enc = tiktoken.encoding_for_model(self.config["model_embeddings"])
        descriptions = list(df["Descripción"])
        total_tokens = sum([len(enc.encode(item)) for item in descriptions])

        for ntok in descriptions:
            val = enc.encode(ntok)
            print(f"\n {ntok}")
            print(val)

        print(f"\n Tokens totales: {total_tokens}")
        cost = total_tokens * (0.13 / 1000000)
        print(f"Costo estimado en USD: {cost:.10f}")

    def create_embeddings(self, text):
        text = text.replace("\n", " ")
        return (
            self.client.embeddings.create(
                input=[text], model=self.config["model_embeddings"]
            )
            .data[0]
            .embedding
        )

    def save_embeddings(self, df, file_name):
        df["embeddings"] = df["Descripción"].apply(lambda x: self.create_embeddings(x))
        df.to_csv(file_name)

    def generate_text(self, prompt):
        """Generar texto con el modelo LLM GTP-4o-mini"""

        message = self.client.chat.completions.create(
            model=self.config["model"],
            max_tokens=self.config["max_tokens"],
            temperature=self.config["temperature"],
            messages=[
                {"role": "system", "content": "You are a smart AI assitant"},
                {"role": "user", "content": prompt},
            ],
        )
        return message.choices[0].message.content


if __name__ == "__main__":
    load_dotenv()

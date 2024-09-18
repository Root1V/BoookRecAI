from openai import OpenAI
from dotenv import load_dotenv
from app.config import Config
from scipy.spatial.distance import cosine
import pandas as pd
import numpy as np
import os
import tiktoken
import ast

class App:

    def __init__(self) -> None:
        self.STORAGE_EMBEDDINGS="data/embeddings.csv"
        self.config = Config.get_all()
        self.client = OpenAI()

    def load_data(self):
        df = pd.read_csv("data/book.csv")
        df = df.dropna(subset=["Descripción"]).reset_index(drop=True)
        # df = df.sample(n=20, random_state=1)
        print(df)
        self.get_coste_embeddings(df)
        return df

    def once_run(self):
        df = self.load_data()
        self.save_embeddings(df, self.STORAGE_EMBEDDINGS)

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

    def load_embeddings(self):
        df_embeddings = pd.read_csv(self.STORAGE_EMBEDDINGS)
        df_embeddings["embeddings"] = df_embeddings["embeddings"].apply(ast.literal_eval).apply(np.array) 
        print(df_embeddings)
        return df_embeddings

    def get_recomendation(self, title, k):
        """
            Obtenemos la similitud por coseno de los vectores.
        """

        df = self.load_embeddings()

        if title not in list(df["Titulo"]):
            return False
        
        # Buscamos el titulo solicitado y otenemos el embedding asociado, si hay mas de un resultado tomamos el primero
        book_embeding = df[df["Titulo"] == title]["embeddings"].iloc[0]
        #si el embedding tiene un solo valor o tiene una dimension de mas los eliminamos
        book_embeding = book_embeding.squeeze()

        embeddings = list(df["embeddings"])
        distances = [cosine(book_embeding, emb) for emb in embeddings]
        # Devuelve los índices de los elementos en distances en orden ascendente (del más pequeño al más grande).
        closest_index = np.argsort(distances)[:k]
        recomendations = list()
        for ix in closest_index[1: k+1]:
            book = dict()
            book["Title"]= df.iloc[ix]["Titulo"]
            book["Description"] = df.iloc[ix]["Descripción"]
            book["Distance"] = distances[ix]
            recomendations.append(book)

        return recomendations
    
    def run(self):
        title = input("Ingrese el titulo del libro:")
        book_recomendations = self.get_recomendation(title, 5)

        if book_recomendations:
            for i, item in enumerate(book_recomendations):
                print(f"\nBook Recomendation #{i+1}, Distance: {item['Distance']}")
                print(f"Titulo: {item['Title']}")
                print(f"Descripcion: {item['Description']}")
                print("_" * 50)
                

if __name__ == "__main__":
    load_dotenv()

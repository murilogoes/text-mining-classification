from fastapi import FastAPI,File, UploadFile
from starlette.requests import Request
from pydantic import BaseModel
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

import pandas as pd
from io import StringIO


import pickle
import csv
import codecs

from utils.CleanText import CleanText

app = FastAPI()

origins = [
    "http://localhost:4200",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextoPredicao(BaseModel):
    texto: str


# @app.post("/items/")
# async def create_item(item: Item):
#     return item


@app.post("/predict/")
async def predicao_texto(predicao: TextoPredicao):
    clean_documents = []

    clean_text = CleanText()

    clean_documents.append(clean_text.clean(predicao.texto))

    # criando o vetor tf-idf

    with open('data/vectorizer/vectorizer_tfidf.pickle', 'rb') as f:
        vectorizer = pickle.load(f)
        X = vectorizer.transform(clean_documents)

    with open('data/trained_models/trained_model.pickle', 'rb') as f:
        ml = pickle.load(f)
        y_pred = ml.predict(X)
        mensagem_retorno = "Contexto Não-Policial" if int(y_pred[0]) == 0 else "Contexto Policial"

    return {"prediction": mensagem_retorno}

@app.post("/lote/")
async def predicao_lote(file: UploadFile = File(...)):
    csv_reader = csv.DictReader(codecs.iterdecode(file.file, 'utf-8-sig'), delimiter=";")
    historicos = []
    for row in csv_reader:
        historicos.append(row['Historico'])


    # csv_reader = csv.DictReader(codecs.iterdecode(file.file, 'utf-8'), delimiter=";")
    # for row in csv_reader:
    #     print(row['Historico'])

    print(file.filename)
    return {"mensagem": "dahora"}

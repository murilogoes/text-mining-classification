from fastapi import FastAPI,File, UploadFile
from fastapi.responses import StreamingResponse
import io
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

import pandas as pd


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


def predict(documents):

    clean_text = CleanText()
    clean_documents = []

    for document in documents:
        clean_documents.append(clean_text.clean(document))

    with open('data/vectorizer/vectorizer_tfidf.pickle', 'rb') as f:
        vectorizer = pickle.load(f)
        X = vectorizer.transform(clean_documents)

    with open('data/trained_models/trained_model.pickle', 'rb') as f:
        ml = pickle.load(f)
        y_pred = ml.predict(X)
        return y_pred

@app.post("/predict/")
async def predicao_texto(predicao: TextoPredicao):
    documents = []
    documents.append(predicao.texto)
    y_pred = predict(documents)
    mensagem_retorno = "Contexto Não-Policial" if int(y_pred[0]) == 0 else "Contexto Policial"
    return {"prediction": mensagem_retorno}

@app.post("/lote/")
async def predicao_lote(file: UploadFile = File(...)):

    csv_arquivo = codecs.iterdecode(file.file, 'utf-8-sig')
    csv_reader = csv.DictReader(csv_arquivo, delimiter=";")

    historicos = []
    for row in csv_reader:
        historicos.append(row['Historico'])

    predictions = predict(historicos)

    df = pd.DataFrame({"Historico": historicos, "Contexto": predictions})
    stream = io.StringIO()
    df.to_csv(stream, index=False,  encoding="utf-8", sep=";")

    response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=export.csv"

    return response

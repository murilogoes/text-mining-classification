import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from utils.CleanText import CleanText
from utils.ConfusionMatrix import ConfusionMatrix
from utils.Tester import Tester

from sklearn.metrics import precision_score, recall_score, f1_score

import pickle

clean_documents = []
labels = []

# # lendo o arquivo csv ja limpo e tratado e adicionando em um array com cada texto e sua respectiva classificacao
# with open('data/todos_bos_tratados_marco.csv','r') as csvfile:
#     next(csvfile)
#     reader = csv.reader(csvfile, delimiter =',')
#     for row in reader:
#         clean_documents.append(row[0])
#         labels.append(row[1])



documents = [
"Comparecem nesta repartição os policiais Tenente Murilo Goes e Soldado Anibal relatando que estavam em patrulhamento de rotina, momento que foram solucionados por populares, informando que havia uma pessoa caída ao solo dentro de um bar na Viela oito, já em óbito por golpes de faca. Consta que Joao Alberto estava jogando cartas com um individuo desconhecido, quando entraram em desentendimento e Joao Alberto foi agredido com golpes de faca, sem possibilidade de socorro. O individuo desconhecido se evadiu, não sendo possível a identificação",
"Comparecem neste Distrito o Cabo Alexandre e Soldado Joao relatando que estavam em atendimento de ocorrência de roubo em andamento a residência na Avenida dos Bacanas 125, momento que avistaram um individuo saindo da residência com arma em punho, sendo que ao avistarem os policiais, efetuou disparos de arma de fogo, sendo devidamente revidado pela guarnição, que atingiu o individuo com cinco disparos de arma de fogo. O individuo foi socorrido, mas entrou em óbito na chegada ao PS Vila Albertina. "
]

clean_documents = []

clean_text = CleanText()

for doc in documents:
    clean_documents.append(clean_text.clean(doc))

# criando o vetor tf-idf

with open('data/vectorizer/vectorizer_tfidf.pickle', 'rb') as f:
    vectorizer = pickle.load(f)
    X = vectorizer.transform(clean_documents)

with open('data/trained_models/trained_model.pickle', 'rb') as f:
    ml = pickle.load(f)
    y_pred = ml.predict(X)
    print(y_pred)

#y_pred = ?.predict(X_test)
# with open('data/trained_models/trained_model.pickle', 'rb') as f:
#     ml = pickle.load(f)
#     y_pred = ml.predict(X)
#
#     test = Tester()
#     #test.test_split()
#
#
#     cnf_matrix = confusion_matrix(labels, y_pred)
#     np.set_printoptions(precision=2)
#
#     print("Acuracia: %s" % ml.score(X, labels))
#     print("F1: %s" % str(f1_score(labels, y_pred, average=None)))
#     print("Recall: %s" % str(recall_score(labels, y_pred, average=None)))
#     print("Precision: %s" % str(precision_score(labels, y_pred, average=None)))
#
#     plt.figure()
#     cfs_matrix = ConfusionMatrix()
#     cfs_matrix.plot_confusion_matrix(cnf_matrix, classes=[0,1], title='Matriz de Confusao')
#
#     plt.show()

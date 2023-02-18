import csv

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from utils.BM25Vectorizer import BM25Vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from imblearn.over_sampling import RandomOverSampler

import pickle

from utils.Tester import Tester


# funcao de treino, vai receber o algoritmo, X, y e o nome das classes (eu separei como 0 e 1)
def train(classifier, X, y, class_names):
    # esse train_test_split eu estou separando minha base em base de treino e base de teste
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    ##TREINANDO MEU ALGORITMO, estou passando a minha amostra de treino
    #trained_model = classifier.fit(X_train, y_train)
    trained_model = classifier.fit(X, y)

    # salvando o modelo
    with open('data/trained_models/trained_model.pickle', 'wb') as f:
        pickle.dump(trained_model, f)

    # aplicando os testes do treino
    #tester = Tester()
    ##Predicoes para medição da Acurácia
    #y_pred = trained_model.predict(X_test)
    #tester.test_split(classifier, X_test, y_test, y_pred, class_names)

    # fazendo cross validation
    # tester.cross_validation(classifier,X, y)
    return classifier


documents = []
labels = []

# lendo o arquivo csv ja limpo e tratado e adicionando em um array com cada texto e sua respectiva classificacao
with open('data/todos_bos_contextos_tratados.csv', 'r', encoding = "ISO-8859-1") as csvfile:
    next(csvfile)
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        documents.append(row[0])
        labels.append(row[1])

# criando o vetor tf-idf
#vectorizer_tfidf = TfidfVectorizer(use_idf=True)
vectorizer_tfidf = TfidfVectorizer(min_df=5, max_df=0.8)
X = vectorizer_tfidf.fit_transform(documents)
y = labels
with open('data/vectorizer/vectorizer_tfidf.pickle', 'wb') as f:
    pickle.dump(vectorizer_tfidf, f)

# BM25
# vectorizer_bm25 = BM25Vectorizer()
# X = vectorizer_bm25.fit_transform(documents)
# y = labels
#
# with open('data/vectorizer/vectorizer_bm25.pickle', 'wb') as f:
#     pickle.dump(vectorizer_bm25, f)

# aqui eu estou fazendo um undersampler para balancear o dataset
# rs = RandomUnderSampler()
# trocando para oversampler com rain forest ficou super bom
rs = RandomOverSampler()
X, y = rs.fit_resample(X, y)

# mandando treinar

# Naive Bayes
# https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html
# testei o alpha em diversos valores, o 0.5 ficou com uma precisao melhor (0.81)
#train(MultinomialNB(alpha=0.5), X, y, [0, 1])

# Linear SVC
# acuracia 0.80 nas esta acertando mais o 1 ... aumentando a interacao para 10mil chegou as vezes a 0.81
# train(svm.LinearSVC(max_iter=10000), X,y, [0,1])

# Random Forest
# parece que aprende mais sobre a classe 0 e a 1 erra muito mas quando eu fiz o oversampler melhorou demais
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
train(RandomForestClassifier(n_estimators=100), X, y, [0, 1])

# Stochastic Gradient Descent Classifier
# train(SGDClassifier(max_iter=1000, tol=1e-3), X,y, [0,1])

# multilayer perceptron demora muito
# train(MLPClassifier(random_state=1, max_iter=300), X,y, [0,1])

# SVC demora muito
# train(svm.SVC(kernel='linear', C=1.0), X,y, [0,1])

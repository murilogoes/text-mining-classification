import csv
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier


from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score

from imblearn.under_sampling import RandomUnderSampler

import pickle
from ConfusionMatrix import ConfusionMatrix


#funcao de treino, vai receber o algoritmo, X, y e o nome das classes (eu separei como 0 e 1)
def train(classifier, X, y, class_names):

    # esse train_test_split eu estou separando minha base em base de treino e base de teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    ##TREINANDO MEU ALGORITMO, estou passando a minha amostra de treino
    trained_model = classifier.fit(X_train, y_train)

    with open('data/trained_models/trained_model.pickle', 'wb') as f:
        pickle.dump(trained_model, f)

    ##Predicoes para medição da Acurácia
    y_pred = trained_model.predict(X_test)

    # na matriz de confusao eu consigo ter uma estimativa de quanto acertou/errou
    cnf_matrix = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)

    print("Acuracia: %s" % classifier.score(X_test, y_test))
    print("F1: %s" % str(f1_score(y_test, y_pred, average=None)))
    print("Recall: %s" % str(recall_score(y_test, y_pred, average=None)))
    print("Precision: %s" % str(precision_score(y_test, y_pred, average=None)))

    plt.figure()
    cfs_matrix = ConfusionMatrix()
    cfs_matrix.plot_confusion_matrix(cnf_matrix, classes=class_names, title='Matriz de Confusao')

    plt.show()

    return classifier

documents = []
labels = []

# lendo o arquivo csv ja limpo e tratado e adicionando em um array com cada texto e sua respectiva classificacao
with open('data/todos_bos_contextos_tratados.csv','r') as csvfile:
    next(csvfile)
    reader = csv.reader(csvfile, delimiter =',')
    for row in reader:
        documents.append(row[0])
        labels.append(row[1])

# criando o vetor tf-idf
vectorizer = TfidfVectorizer(use_idf=True)
X = vectorizer.fit_transform(documents)
y = labels

with open('data/vectorizer/vectorizer.pickle', 'wb') as f:
    pickle.dump(vectorizer, f)

# aqui eu estou fazendo um undersampler para balancear o dataset
rus = RandomUnderSampler()
X, y = rus.fit_resample(X, y)


#mandando treinar

# Naive Bayes
#train(MultinomialNB(alpha=0.05), X, y, [0,1])

# SVC
#train(svm.SVC(kernel='linear', C=1.0), X,y, [0,1])

# Linear SVC
#train(svm.LinearSVC(random_state=0, tol=1e-05), X,y, [0,1])

# Random Forest
#train(RandomForestClassifier(max_depth=2, random_state=0), X,y, [0,1])

# Stochastic Gradient Descent Classifier
train(SGDClassifier(max_iter=1000, tol=1e-3), X,y, [0,1])

# multilayer perceptron
#train(MLPClassifier(random_state=1, max_iter=300), X,y, [0,1])


#tentar linearSvc SGDClassifier Random Forest  Multilayer Perceprtron (MLP)




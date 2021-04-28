import csv
import numpy as np
import matplotlib.pyplot as plt
import itertools

from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score

# codigo para criar a matriz de confusao pego da documentacao do scikit
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

#funcao de treino, vai receber o algoritmo, X, y e o nome das classes (eu separei como 0 e 1)
def train(classifier, X, y, class_names):

    # esse train_test_split eu estou separando minha base em base de treino e base de teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    ##TREINANDO MEU ALGORITMO, estou passando a minha amostra de treino
    classifier.fit(X_train, y_train)

    ##Predicoes para medição da Acurácia
    y_pred = classifier.fit(X_train, y_train).predict(X_test)

    # na matriz de confusao eu consigo ter uma estimativa de quanto acertou/errou
    cnf_matrix = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)

    print("Acuracia: %s" % classifier.score(X_test, y_test))
    print("F1: %s" % str(f1_score(y_test, y_pred, average=None)))
    print("Recall: %s" % str(recall_score(y_test, y_pred, average=None)))
    print("Precision: %s" % str(precision_score(y_test, y_pred, average=None)))

    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, title='Matriz de Confusao')

    plt.show()

    return classifier

documents = []
labels = []

# lendo o arquivo csv ja limpo e tratado e adicionando em um array com cada texto e sua respectiva classificacao
with open('data/bos2.csv','r') as csvfile:
    next(csvfile)
    reader = csv.reader(csvfile, delimiter =',')
    for row in reader:
        documents.append(row[0])
        labels.append(row[1])

# criando o vetor tf-idf
vectorizer = TfidfVectorizer(use_idf=True)
X = vectorizer.fit_transform(documents)
y = labels

#mandando treinar
#train(MultinomialNB(alpha=0.05), X, y, [0,1])
train(svm.SVC(kernel='linear', C=1.0), X,y, [0,1])
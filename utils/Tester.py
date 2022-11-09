from utils.ConfusionMatrix import ConfusionMatrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import LabelBinarizer
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt


class Tester:

    def test_split(self, classifier, X_test, y_test, y_pred, class_names):
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

    def cross_validation(self, classifier, X, y):
        # apliando cross validation
        lb = LabelBinarizer()
        y = np.array([number[0] for number in lb.fit_transform(y)])

        #cv_results = cross_val_score(classifier, X, y, cv=5, scoring='f1')
        cv_results = cross_validate(classifier, X, y, cv=10,
                                    scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'])
        cv_results_df = pd.DataFrame(cv_results)

        print("=== All AUC Scores ===")
        print(cv_results)
        print('\n')
        print("=== Mean AUC Score ===")
        print("Mean AUC Score - Random Forest: ", cv_results_df.mean())

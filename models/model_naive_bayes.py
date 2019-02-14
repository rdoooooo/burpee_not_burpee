import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from general_tools import clean_data, ROC, bar_coef, cm


X_train_scaled, y_train, X_test_scaled, y_test = clean_data(scaled=1)


gaussianNB = GaussianNB()
bernoulliNB = BernoulliNB()
multinomialNB = MultinomialNB()

for nb in [gaussianNB, bernoulliNB, multinomialNB]:
    print(f'Working {str(nb)}\n')
    nb.fit(X_train_scaled, y_train)
    y_pred = nb.predict(X_test_scaled)
    report = classification_report(y_true=y_test, y_pred=y_pred)
    print(report)
    print(confusion_matrix(y_true=y_test, y_pred=y_pred))

    # Confusion matrix
    CM = confusion_matrix(y_true=y_test, y_pred=y_pred, labels=[0, 1])
    cm(cm=CM, target_names=['Burpee no jump', 'Burpee'],
       title=str(nb) + ' CM')

    # ROC
    ROC(Model=nb, Y_test=y_test, X_test=X_test_scaled)

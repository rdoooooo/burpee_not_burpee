import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from general_tools import clean_data, ROC, bar_coef, cm


def run_model():
    # Load data
    X_train_scaled, y_train, X_test_scaled, y_test = clean_data(scaled=1)

    # Create logistic regresion
    logistic = LogisticRegression(
        solver='liblinear', max_iter=150, random_state=42)

    # Create regularization penalty space
    penalty = ['l1', 'l2']
    # Create regularization hyperparameter space
    # First run in logspace found C to be ~10
    C = np.linspace(8, 12, 30)
    # Create hyperparameter options
    hyperparameters = dict(C=C, penalty=penalty)

    # Create grid search using 5 split stratisfied shuffle split cross validation
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    clf = GridSearchCV(logistic, hyperparameters, cv=cv,
                       verbose=0, scoring='precision', n_jobs=-1)

    # Fit grid search
    best_model = clf.fit(X_train_scaled, y_train)

    # View best hyperparameters
    print('Best Penalty:', best_model.best_estimator_.get_params()['penalty'])
    print('Best C:', best_model.best_estimator_.get_params()['C'])
    print(f'Best training score {best_model.best_score_}')

    # Make a prediction on entire training set
    y_pred = best_model.best_estimator_.predict(X_test_scaled)

    # Classification report showing precision,
    report = classification_report(y_true=y_test, y_pred=y_pred)
    print(report)

    # Display confusion matrix
    CM = confusion_matrix(y_true=y_test, y_pred=y_pred, labels=[0, 1])
    cm(cm=CM, target_names=['Burpee no jump', 'Burpee'],
       title='Logistric Regression CM')

    # Plot a bar graph of the variables to get insight in importance
    bar_coef(Model=best_model, X_train_scaled=X_test_scaled)

    # Plot ROC curve
    ROC(Model=best_model.best_estimator_, Y_test=y_test, X_test=X_test_scaled)

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from general_tools import clean_data, ROC, bar_coef, cm


def run_model():
    X_train_scaled, y_train, X_test_scaled, y_test = clean_data(scaled=1)

    # Create knn
    knn = KNeighborsClassifier()

    # Create n neighbors hyperparameter space
    n_neighbors = np.arange(start=1, stop=30, step=1)
    # Create hyperparameter options
    hyperparameters = dict(n_neighbors=n_neighbors)

    # Create grid search using 5 split stratisfied shuffle split cross validation
    cv = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
    clf = GridSearchCV(knn, hyperparameters, cv=cv, verbose=0,
                       scoring='precision', n_jobs=-1)

    # Fit grid search
    best_model = clf.fit(X_train_scaled, y_train)

    # View best hyperparameters
    print('Best N:', best_model.best_estimator_.get_params()['n_neighbors'])
    print(f'Train score is {best_model.best_score_}')

    cv_scores = best_model.cv_results_['mean_test_score']
    plt.plot(n_neighbors, cv_scores)
    plt.xlabel('K'), plt.ylabel('Mean Test Score - Precision')
    # Make a prediction on entire training set
    y_pred = best_model.best_estimator_.predict(X_test_scaled)

    # Classification Report
    report = classification_report(y_true=y_test, y_pred=y_pred)
    print(report)

    # Confusion Matrix
    CM = confusion_matrix(y_true=y_test, y_pred=y_pred, labels=[0, 1])
    cm(cm=CM, target_names=['Burpee no jump', 'Burpee'],
       title='KNN CM')

    # ROPC
    ROC(Model=best_model.best_estimator_, Y_test=y_test, X_test=X_test_scaled)

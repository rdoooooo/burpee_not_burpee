import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from general_tools import clean_data, ROC, bar_coef, cm


X_train_scaled, y_train, X_test_scaled, y_test = clean_data(scaled=1)

# Create SVC
svc = SVC(max_iter=-1, probability=True)


# Create diff kernels
kernel = ['rbf', 'linear', 'poly', 'sigmoid']

# Degree for poly kernels
degree = np.arange(start=1, stop=5, step=1)
# Gamma for poly kernels
gamma = np.logspace(start=-15, stop=4, num=18, base=2)
# C penalty factor
C = np.logspace(start=-3, stop=16, num=18, base=2)

# Create hyperparameter options
hyperparameters = dict(kernel=kernel, degree=degree, gamma=gamma, C=C)

# Create grid search using 5 split stratisfied shuffle split cross validation
cv = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
clf = GridSearchCV(svc, hyperparameters, cv=cv, verbose=1,
                   scoring='precision', n_jobs=-1)

# Fit grid search
best_model = clf.fit(X_train_scaled, y_train)

# View best hyperparameters
print(f'Best parameters {best_model.best_params_}')
print(f'Train score is {best_model.best_score_}')

# Best parameters {'C': 0.125, 'degree': 2, 'gamma': 0.07063223646433309, 'kernel': 'poly'}

# Make a prediction on entire training set
y_pred = best_model.best_estimator_.predict(X_test_scaled)

# Classification report
report = classification_report(y_true=y_test, y_pred=y_pred)
print(report)

# Confusion Matrix
CM = confusion_matrix(y_true=y_test, y_pred=y_pred, labels=[0, 1])
cm(cm=CM, target_names=['Burpee no jump', 'Burpee'],
   title='SVM CM')

# ROC
ROC(Model=best_model.best_estimator_, Y_test=y_test, X_test=X_test_scaled)

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from general_tools import clean_data, ROC, bar_coef, cm


X_train_scaled, y_train, X_test_scaled, y_test = clean_data(scaled=1)
# Create knn
rf = RandomForestClassifier(random_state=42)


# Hyperparameter space for RF
# Number of trees in random forest
n_estimators = n_estimators = np.arange(start=1, stop=30, step=1)
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

# Create grid search using 5 split stratisfied shuffle split cross validation
cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
clf = rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid,
                                     n_iter=100, cv=3, verbose=2, random_state=42, n_jobs=-1, scoring='precision')

# Fit grid search
best_model = clf.fit(X_train_scaled, y_train)

# View best hyperparameters
print(f'Best parameters {best_model.best_params_}')
print(f'Train score is {best_model.best_score_}')
# Best parameters {'n_estimators': 17, 'min_samples_split': 10, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'max_depth': 80, 'bootstrap': False}
# Train score is 0.9477959255356516


# Make a prediction on entire training set
y_pred = best_model.best_estimator_.predict(X_test_scaled)
report = classification_report(y_true=y_test, y_pred=y_pred)
print(report)

# Confusion Matrix
CM = confusion_matrix(y_true=y_test, y_pred=y_pred, labels=[0, 1])
cm(cm=CM, target_names=['Burpee no jump', 'Burpee'],
   title='Random Forest CM')

# ROC
ROC(Model=best_model.best_estimator_, Y_test=y_test, X_test=X_test_scaled)

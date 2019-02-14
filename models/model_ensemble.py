

from sklearn import linear_model, svm, naive_bayes, neighbors, ensemble
df = pd.read_csv('data/dataframe.csv')
X_train, X_test, y_train, y_test = model_selection.train_test_split(
    df.drop('label', axis=1), df.label, random_state=123)


lr_model = linear_model.LogisticRegression()
nb_model = naive_bayes.GaussianNB()
knn_model = neighbors.KNeighborsClassifier()
svc_model = svm.SVC(probability=True, gamma="scale")
rf_model = ensemble.RandomForestClassifier(n_estimators=100)
et_model = ensemble.ExtraTreesClassifier(n_estimators=100)
ada_model = ensemble.AdaBoostClassifier()

models = ["lr_model", "nb_model", "knn_model",
          "svc_model", "rf_model", "et_model", "ada_model"]

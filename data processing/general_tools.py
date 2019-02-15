'''
Holds various tools called by other scripts. Tools include the following:
-Confusion matrix : cm
-Load data - features into X and y for model: load_data
-Min max data scaler: scale_data. Min max used because of data distribution
-Split train/test data input into models: clean_data
-Generate a ROC curve: ROC
-Generate a bar plot to visualize coefficients for LR: bar_coef
-Filter data (high/low/bandpass):butter_[]_filter
'''

from scipy.signal import butter, filtfilt
from rfpimp import importances, plot_importances, RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


def cm(cm,
       target_names,
       title='Confusion matrix',
       cmap=None,
       normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(
        accuracy, misclass))
    plt.show()


####################

current_path = '/Users/richarddo/Documents/github/Metis/Projects/burpee_not_burpee/'


def load_data(loc='data/df_train.pk', current_path=current_path):

    df = pd.read_pickle(current_path + loc)
    df.sort_index(inplace=True)
    df.rename({'Truth': 'label'}, axis=1, inplace=True)
    df.label.astype(np.int, inplace=True)
    X = df.drop('label', axis=1)
    y = df.label
    return X, y


def scale_data(X, y, scaled=1):

    # Load data
    X_train, y_train = load_data()
    X_test, y_test = load_data(loc='/data/df_test.pk')

    # Do a min max scaler on all the featuers
    scaler = MinMaxScaler()
    X_train_scaled = pd.DataFrame(data=scaler.fit_transform(
        X_train), columns=X_train.columns.tolist())
    X_test_scaled = pd.DataFrame(data=scaler.fit_transform(
        X_test), columns=X_test.columns.tolist())
    if scaled:
        return X_train_scaled, y_train, X_test_scaled, y_test
    else:
        return X_train, y_train, X_test, y_test


def clean_data(scaled=1):
    X, y = load_data()
    X_train, y_train, X_test, y_test = scale_data(X, y, scaled=scaled)
    return X_train, y_train, X_test, y_test


def ROC(Model, Y_test, X_test):
    # Perforamnce of the model
    fpr, tpr, _ = roc_curve(Y_test, Model.predict_proba(X_test)[:, 1])
    AUC = auc(fpr, tpr)
    print('the AUC is : %0.4f' % AUC)
    plt.figure(figsize=(12, 5))
    plt.plot(fpr, tpr, label='ROC curve (area = %0.4f)' % AUC)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()


def bar_coef(Model, X_train_scaled):
    plt.figure(figsize=(10, 10))
    plt.grid(b='on')
    plt.barh(y=X_train_scaled.columns,
             width=Model.best_estimator_.coef_[0])
    plt.xlabel('Beta magitude')
    plt.xlabel('Features')


# Builds the filter


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_lowpass(lowcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    b, a = butter(order, low, btype='low')
    return b, a


def butter_highpass(highcut, fs, order=5):
    nyq = 0.5 * fs
    high = highcut / nyq
    b, a = butter(order, high, btype='high')
    return b, a


# Filters signal
def butter_highpass_filter(data, highcut, fs, order=5):
    b, a = butter_highpass(highcut, fs, order)
    y = filtfilt(b, a, data)
    return y


def butter_lowpass_filter(data, lowcut,  fs, order=5):
    b, a = butter_lowpass(lowcut, fs, order)
    y = filtfilt(b, a, data)
    return y


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    y = filtfilt(b, a, data)
    return y

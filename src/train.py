from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, multilabel_confusion_matrix
from src import config
from itertools import cycle
from sklearn import svm
import joblib
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score
import warnings
warnings.filterwarnings('always')

def train(X_data, y_data):
    X = pd.read_csv(X_data)
    y = pd.read_csv(y_data)
    y = y.values.reshape(-1, )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    def roc_aux_plot(model, X, y):
        """
        This  function takes model, X data and y label and returns ROC AUX metrics and draws for each classes
        """

        y = label_binarize(y, classes=[3, 4, 5, 6, 7, 8, 9])
        n_classes = y.shape[1]

        random_state = np.random.RandomState(0)
        n_samples, n_features = X.shape
        X = np.c_[X, random_state.randn(n_samples, n_features)]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

        classifier = OneVsRestClassifier(model)
        y_score = classifier.fit(X_train, y_train).predict_proba(X_test)
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        # Finally average it and compute AUC
        mean_tpr /= n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        print("Our macro average ROC curve is: ", roc_auc["macro"])
        # Plot all ROC curves
        lw = 2
        plt.figure(figsize=(7, 7))
        plt.plot(fpr["micro"], tpr["micro"],
                 label='micro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["micro"]),
                 color='deeppink', linestyle=':', linewidth=4)

        plt.plot(fpr["macro"], tpr["macro"],
                 label='macro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["macro"]),
                 color='navy', linestyle=':', linewidth=4)

        colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'r', 'b', 'brown', 'green'])
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                     label='ROC curve of class {0} (area = {1:0.2f})'
                           ''.format(i + 3, roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC AUX for multi-class by using {model}')
        plt.legend(loc="lower right")
        plt.show()

    def evaluate_model(model, predicions):
        print('The model is ', model)
        result = {"Accuracy": np.round(accuracy_score(predicions, y_test), 3),
                  "Recall": np.round(recall_score(predicions, y_test, average='weighted', labels=np.unique(predicions)), 3),
                  "Precision": np.round(
                      precision_score(predicions, y_test, average='weighted', labels=np.unique(predicions)), 3),
                  "F1_score": np.round(f1_score(y_test, predicions, labels=np.unique(predicions), average='weighted'), 3)
                  }

        return result

    MODELS = {
        'logisticregression': LogisticRegression(random_state=0, solver='liblinear', max_iter=2000),
        'knn': KNeighborsClassifier(),
        'randomforest': RandomForestClassifier(n_estimators=31, random_state=0),
        'svm': svm.SVC(random_state=0, probability=True)
    }
    model = MODELS[config.MODEL]

    if config.MODEL == 'logisticregression':
        param_grid = {'penalty': ['l1', 'l2'], 'C': np.logspace(-4, 4, 20)}
        grid_model = GridSearchCV(model, param_grid=param_grid, cv=5)
        model = grid_model.fit(X_train, y_train).best_estimator_
    if config.MODEL == 'knn':
        param_grid = {'n_neighbors': range(3, 30)}
        grid_model = GridSearchCV(model, param_grid=param_grid, cv=5)
        model = grid_model.fit(X_train, y_train).best_estimator_
    if config.MODEL == 'randomforest':
        param_grid = {'max_depth': range(2, 9), 'min_samples_split': range(2, 11, 2)}
        grid_model = GridSearchCV(model, param_grid=param_grid, cv=5)
        model = grid_model.fit(X_train, y_train).best_estimator_
    if config.MODEL == 'svm':
        param_grid = {'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 'C': np.arange(0.001, 2, 0.2)}
        grid_model = GridSearchCV(model, param_grid=param_grid, cv=5)
        model = grid_model.fit(X_train, y_train).best_estimator_

    y_pred = model.predict(X_test)

    joblib.dump(model, config.MODELS_PATH + f'trained_model_{config.MODEL}.pkl')

    print("Our scores are: ", evaluate_model(config.MODEL, y_pred))
    print("Our multilabel_confusion_matrix is:", multilabel_confusion_matrix(y_test, y_pred), sep="\n")
    print("This is the score for train set: ", model.score(X_train, y_train))
    print("This is the score for test set: ", model.score(X_test, y_test))
    roc_aux_plot(model, X, y)
    plt.figure(figsize=(10, 5))
    plt.barh(X.columns, model.feature_importances_)
    plt.title('Feature importances')
    plt.show()


train(config.PROCESSED_X_H1, config.LABEL_DATA)
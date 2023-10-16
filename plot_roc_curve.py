import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.base import is_classifier
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import check_cv

import pandas as pd

def filter_idxs(v_array, idxs, return_copy=False, convert_to_ndarray=True):

    if convert_to_ndarray:
        if isinstance(v_array, pd.DataFrame):
            if return_copy:
                return v_array.iloc[idxs].copy().values
            else:
                return v_array.iloc[idxs].values

        elif isinstance(v_array, np.ndarray):
            if return_copy:
                return v_array[idxs].copy()
            else:
                return v_array[idxs]

    else:
        if isinstance(v_array, pd.DataFrame):
            if return_copy:
                return v_array.iloc[idxs].copy()
            else:
                return v_array.iloc[idxs]

        elif isinstance(v_array, np.ndarray):
            if return_copy:
                return v_array[idxs].copy()
            else:
                return v_array[idxs]


def plot_roc_curve(X, y,
                   clf,
                   clf_label=None,
                   target_column=1, # for binary is the usual
                   class_name=None,
                   color_mean_roc_curve='r',
                   cv=5, 
                   n_points_roc_curve=200,
                   show_fold_curves=True, show_fold_scores=False, plot_chance_curve=True,
                   figsize=None,
                   dict_pyplot_style=None,
                   fig=None,
                   return_fig=False):


    assert is_classifier(clf), 'clf must be a classifier.'

    if clf_label is None:
        clf_label = ''

    if class_name is None:
        class_name = str(target_column)

    if not figsize:
        figsize = (9, 6)

    kfold = check_cv(cv=cv, y=y, classifier=True)

    score_acc = list()

    # From:
    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html#:~:text=Example%20of%20Receiver%20Operating%20Characteristic,rate%20on%20the%20X%20axis.
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, n_points_roc_curve)

    
    with mpl.rc_context(dict_pyplot_style):

        if fig is not None:
            fig = plt.figure(figsize=figsize)

        for ifold, (train_index, test_index) in enumerate(kfold.split(X, y)):

            X_train = filter_idxs(X, train_index, return_copy=True) # fazer uma copia aqui porque a ideia é poder alterar X_train nessa iteração
            y_train = filter_idxs(y, train_index)

            X_test = filter_idxs(X, test_index)
            y_test = filter_idxs(y, test_index)

            if isinstance(y_train, np.ndarray):
                y_train = y_train.ravel()
                y_test = y_test.ravel() 

            #X_train = X[train_index].copy() # fazer uma copia aqui porque a ideia é poder alterar X_train nessa iteração
            #y_train = y[train_index]

            #X_test = X[test_index]
            #y_test = y[test_index]

            # I can process X_train here (standardization for example...)
            #
            #

            # Now I'll train

            clf.fit(X_train, y_train)

            y_true = y_test
            y_pred = clf.predict(X_test)
            y_pred_proba = clf.predict_proba(X_test)[:, target_column]

            # Roc curve
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)

            auc_ = roc_auc_score(y_true, y_pred_proba)

            if show_fold_curves:
                if show_fold_scores:
                    plt.plot(fpr, tpr, '-', c='gray', label=f'AUC fold {ifold+1}: {auc_:.2f}', figure=fig)
                else:
                    plt.plot(fpr, tpr, '-', c='gray', figure=fig)

            aucs.append(auc_)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        plt.plot(
            mean_fpr,
            mean_tpr,
            color=color_mean_roc_curve,
            label=r"Mean ROC %s for class %s, AUC = %0.2f $\pm$ %0.2f" % (clf_label, class_name, mean_auc, std_auc),
            lw=4,
            alpha=0.8,
            figure=fig
        )
        if plot_chance_curve:
            plt.plot([0, 1], [0, 1], '--', label='chance', lw=4, figure=fig)
        plt.legend()
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')

        if return_fig:
            return fig
        else:
            plt.show()



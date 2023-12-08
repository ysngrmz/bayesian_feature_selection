import numpy as np

import random

from skopt import gp_minimize
from skopt.space import Categorical
from skopt.utils import use_named_args

from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


models_dict = {
    "svm": svm.SVC(decision_function_shape='ovo'),
    "knn": KNeighborsClassifier(n_neighbors=3),
    "gaussianNB": GaussianNB(),
    "multinominalNB": MultinomialNB(force_alpha=True),
    "logisticregression": LogisticRegression(random_state=0),
    "randomforest": RandomForestClassifier(max_depth=2, random_state=0),
    "bagging": BaggingClassifier(random_state=0),
    "adaboost": AdaBoostClassifier(random_state=0),
    "decisiontree": DecisionTreeClassifier(random_state=0),
}

metrics_dic = {
    "acc": accuracy_score,
    "precision": precision_score,
    "recall":recall_score,
    "f1":f1_score
}

best_metric_score = 0.0
best_feature_inds = []
score_and_feature_matrix = []

def call(X,
         y,
         model_name="knn",
         metric_name="acc",
         val_percentage = .2,
         n_calls = 10,
         acq_func = 'EI'):

    number_of_feature = np.shape(X)[1]
    param_grid = [Categorical([True, False], name=str(i)) for i in range(number_of_feature)]

    global best_metric_score, best_feature_inds, score_and_feature_matrix

    best_metric_score = 0.0
    best_feature_inds = []
    score_and_feature_matrix = []

    @use_named_args(dimensions=param_grid)
    def feature_selection_methods(**kwargs):

        global best_metric_score, best_feature_inds, score_and_feature_matrix

        feature_inds = [elem for elem in kwargs.values()]

        inputs_ = X[:, feature_inds]
        number_of_sample = X.shape[0]
        number_of_val = int(number_of_sample * val_percentage)

        val_inds = random.sample(range(0, number_of_sample), number_of_val)
        train_inds = [elem for elem in range(number_of_sample) if elem not in val_inds]

        train_inputs = inputs_[train_inds, :]
        train_targets = y[train_inds]
        val_inputs = inputs_[val_inds, :]
        val_targets = y[val_inds]

        model = models_dict[model_name]
        metric = metrics_dic[metric_name]

        model.fit(train_inputs, train_targets)
        preds = model.predict(val_inputs)
        if metric == "acc":
            score = metric(val_targets, preds)
        else:
            score = metric(val_targets, preds,average='macro')

        if score > best_metric_score:
            best_metric_score = score
            best_feature_inds = feature_inds

        temp_score_and_feature_matrix = [score]
        for elem in feature_inds:
            temp_score_and_feature_matrix.append(elem)

        score_and_feature_matrix.append(temp_score_and_feature_matrix)

        return -score

    for i in range(1,number_of_feature+1):
        setattr(feature_selection_methods,str(i),None)
    gp_minimize(feature_selection_methods, dimensions=param_grid,acq_func=acq_func, n_calls=n_calls)

    return best_metric_score, best_feature_inds, score_and_feature_matrix
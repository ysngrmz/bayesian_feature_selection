import numpy as np

import bayesian_feature_selection as bfs

import sklearn.datasets as ds
from sklearn.model_selection import train_test_split

from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from tabulate import tabulate

from sklearn.metrics import accuracy_score, precision_score, recall_score

import copy

test_percentage = 0.25

#Create train test for iris dataset
iris_data = ds.load_iris()

X_train_iris, X_test_iris, y_train_iris, y_test_iris = train_test_split(iris_data.data, iris_data.target,
                                                    stratify=iris_data.target,
                                                    test_size=test_percentage)

#Create train test for digit dataset
digits_data = ds.load_digits()

X_train_digits, X_test_digits, y_train_digits, y_test_digits = train_test_split(digits_data.data, digits_data.target,
                                                    stratify=digits_data.target,
                                                    test_size=test_percentage)


#Create train test for breast cancer wisconsin dataset
breast_data = ds.load_breast_cancer()

X_train_breast, X_test_breast, y_train_breast, y_test_breast = train_test_split(breast_data.data, breast_data.target,
                                                    stratify=breast_data.target,
                                                    test_size=test_percentage)

print(X_train_iris.shape, X_test_iris.shape)
print(X_train_digits.shape, X_test_digits.shape)
print(X_train_breast.shape, X_test_breast.shape)

uniqueTrTarget, countsTrTarget = np.unique(y_train_iris, return_counts=True)
print(uniqueTrTarget)
uniqueTrTarget, countsTrTarget = np.unique(y_train_digits, return_counts=True)
print(uniqueTrTarget)
uniqueTrTarget, countsTrTarget = np.unique(y_train_breast, return_counts=True)
print(uniqueTrTarget)

dataset_list = [
        {"ds_name":"iris",
         "x_train":X_train_iris,
         "x_test":X_test_iris,
         "y_train":y_train_iris,
         "y_test":y_test_iris},

        {"ds_name":"digits",
         "x_train":X_train_digits,
         "x_test":X_test_digits,
         "y_train":y_train_digits,
         "y_test":y_test_digits},

        {"ds_name":"breast",
         "x_train":X_train_breast,
         "x_test":X_test_breast,
         "y_train":y_train_breast,
         "y_test":y_test_breast},
    ]

models_dict = {
    "svm": svm.SVC(decision_function_shape='ovo', probability=True),
    "knn": KNeighborsClassifier(n_neighbors=3),
    "gaussianNB": GaussianNB(),
    "multinominalNB": MultinomialNB(force_alpha=True),
    "logisticregression": LogisticRegression(random_state=0, solver="liblinear"),
    "randomforest": RandomForestClassifier(max_depth=2, random_state=0),
    "bagging": BaggingClassifier(random_state=0),
    "adaboost": AdaBoostClassifier(random_state=0),
    "decisiontree": DecisionTreeClassifier(random_state=0),
}



for ds_ in dataset_list:
    print(ds_["ds_name"])
    best_inds = []
    result_fs = [["Model Name", "Accuracy", "Precision", "Recall"]]
    result_full_feature = [["Model Name", "Accuracy", "Precision", "Recall"]]

    np.savetxt("preds_targets/" + ds_["ds_name"] + "_" + "targets.txt", ds_["y_test"])
    for model_name, model in models_dict.items():

        model_full_feature = copy.copy(model)

        bmr, bfi, sfm= bfs.call(ds_["x_train"],
                                ds_["y_train"],
                                model_name=model_name,
                                metric_name="acc",
                                n_calls=10)

        x_train = ds_["x_train"][:, bfi]
        x_test = ds_["x_test"][:, bfi]
        best_inds.append([model_name, bfi])

        model.fit(x_train, ds_["y_train"])
        preds_fs = model.predict(x_test)
        preds_fs_proba = model.predict_proba(x_test)
        acc_fs = accuracy_score(ds_["y_test"], preds_fs)
        precision_fs = precision_score(ds_["y_test"], preds_fs, average="macro")
        recall_fs = recall_score(ds_["y_test"], preds_fs, average="macro")
        result_fs.append([model_name, acc_fs, precision_fs, recall_fs])

        model_full_feature.fit(ds_["x_train"], ds_["y_train"])
        preds_full_feature = model_full_feature.predict(ds_["x_test"])
        preds_full_feature_proba = model_full_feature.predict_proba(ds_["x_test"])
        acc_full_feature = accuracy_score(ds_["y_test"], preds_full_feature)
        precision_full_feature = precision_score(ds_["y_test"], preds_full_feature, average="macro")
        recall_full_feature = recall_score(ds_["y_test"], preds_full_feature, average="macro")
        result_full_feature.append([model_name, acc_full_feature, precision_full_feature, recall_full_feature])

        preds_target_write_path = "preds_targets/" + ds_["ds_name"] + "_" + model_name + "_"
        np.savetxt(preds_target_write_path + "preds_fs.txt", preds_fs_proba)
        np.savetxt(preds_target_write_path + "preds_full_feature.txt", preds_full_feature_proba)


    experiment_results_write_path = "experiment_results/" + ds_["ds_name"] + "_"
    fs_result_file = open(experiment_results_write_path + "performace_score_fs.txt", "w")
    full_feature_result_file = open(experiment_results_write_path + "performace_score_full_feature.txt", "w")
    selected_feature_file = open(experiment_results_write_path + "selected_feature_file.txt", "w")

    content_results_fs = tabulate(result_fs)
    fs_result_file.write(content_results_fs)
    fs_result_file.close()

    content_features = tabulate(best_inds)
    selected_feature_file.write(content_features)
    selected_feature_file.close()

    content_results_full_feature = tabulate(result_full_feature)
    full_feature_result_file.write(content_results_full_feature)
    full_feature_result_file.close()


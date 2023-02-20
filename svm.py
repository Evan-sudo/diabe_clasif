import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score
import sklearn.svm as svm
import math
import pandas as pd

train_test_ratio = 0.8

train1 = np.load("./data/train_non.npy") 
label1 = np.load("./data/label_non.npy") 
c = list(zip(train1,label1))
random.shuffle(c)
train1, label1 = zip(*c)
print(np.shape(train1))
print(np.shape(label1))

ratio_len = math.ceil(train_test_ratio*len(train1))
x_train = train1[0:ratio_len]
x_test = train1[ratio_len+1:]
y_train = label1[0:ratio_len]
y_test = label1[ratio_len+1:]

def svc(kernel):
    return svm.SVC(kernel=kernel, decision_function_shape="ovo")


def nusvc():
    return svm.NuSVC(nu=0.5,decision_function_shape="ovo")


def linearsvc():
    return svm.LinearSVC(multi_class="ovr")


def modelist():
    modelist = []
    kernalist = {"linear", "poly", "rbf", "sigmoid"}
    for each in kernalist:
        modelist.append(svc(each))
    #modelist.append(nusvc())
    modelist.append(linearsvc())
    return modelist


def svc_model(model):
    model.fit(x_train, y_train)
    acu_train = model.score(x_train, y_train)
    acu_test = model.score(x_test, y_test)
    y_pred = model.predict(x_test)
    recall = recall_score(y_test, y_pred, average="macro")
    return acu_train, acu_test, recall


def run_svc_model(modelist):
    result = {"kernel": [],
              "acu_train": [],
              "acu_test": [],
              "recall": []
              }

    for model in modelist:
        acu_train, acu_test, recall = svc_model(model)
        try:
            result["kernel"].append(model.kernel)
        except:
            result["kernel"].append(None)
        result["acu_train"].append(acu_train)
        result["acu_test"].append(acu_test)
        result["recall"].append(recall)

    return pd.DataFrame(result)

result = run_svc_model(modelist())
print(result)



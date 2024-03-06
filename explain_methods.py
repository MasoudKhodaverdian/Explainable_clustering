from sklearn.metrics import accuracy_score
from sklearn.inspection import permutation_importance
import numpy as np
from utils import sort_importances
from sklearn.linear_model import LogisticRegression
import progressbar
from sklearn.metrics import accuracy_score
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import f1_score
import random
import copy


def method_1(data,name_columns,cl,lbl,clf=LogisticRegression(random_state=0,penalty='none'),n_repeats=10): # first classification then explain
    # cl is culstering model and it has trained
    # clf is the classification model and it has not trained
    # lbl is labels of the clustering model's output
    clf.fit(data, lbl)
    # print(accuracy_score(lbl,clf.predict(data)))
    r = permutation_importance(clf, data, lbl,n_repeats=n_repeats,random_state=0,n_jobs=-2)
    importances =r.importances_mean
    importances=np.asarray(importances)
    importances=importances/importances.sum()
    feature_importance=dict()
    for i in range(len(importances)):
        feature_importance[name_columns[i]]=importances[i]

    return sort_importances(feature_importance)

def method_2(data,name_columns,cl,lbl,n_repeats=10):
    X=data
    y=lbl
    num_iterations=n_repeats
    clustering_model=cl
    metric = f1_score
    bar=progressbar.ProgressBar(maxval=len(name_columns))
    bar.start()
    baseline_metric=metric(y, clustering_model.predict(X), average='macro')
    scores={c:[] for c in name_columns}
    for c in name_columns:
        X1=X.copy()
        for _ in range(num_iterations):
            temp=list(X1[name_columns.index(c)])
            random.shuffle(temp)
            X1[name_columns.index(c)]=temp
            score=metric(y, clustering_model.predict(X1), average='macro')
            scores[c].append(abs(baseline_metric-score))
        bar.update(name_columns.index(c)+1)
    importances=[]
    for x in scores.values():
        importances.append(np.asarray(x).mean())


    importances=np.asarray(importances)
    importances= importances/importances.sum()
    feature_importance=dict()
    for i in range(len(importances)):
        feature_importance[name_columns[i]]=importances[i]

    return sort_importances(feature_importance)

def method_3(data,name_columns,cl,lbl,n_repeats=10):
    main_cl = copy.deepcopy(cl)
    X=data
    y=lbl
    num_iterations=n_repeats
    metric= adjusted_rand_score
    bar=progressbar.ProgressBar(maxval=len(name_columns))
    bar.start()
    cl = copy.deepcopy(main_cl)
    t_lbl = cl.fit_predict(X)
    baseline_metric=metric(y,t_lbl )
    scores={c:[] for c in name_columns}
    for c in name_columns:
        X1=X.copy()
        for _ in range(num_iterations):
            temp=list(X1[name_columns.index(c)])
            random.shuffle(temp)
            X1[name_columns.index(c)]=temp
            cl = copy.deepcopy(main_cl)
            t_lbl = cl.fit_predict(X1)
            score=metric(y, t_lbl)
            scores[c].append(abs(baseline_metric-score))
        bar.update(name_columns.index(c)+1)

    importances=[]
    for x in scores.values():
        importances.append(np.asarray(x).mean())


    importances=np.asarray(importances)
    importances= importances/importances.sum()
    feature_importance=dict()
    for i in range(len(importances)):
        feature_importance[name_columns[i]]=importances[i]

    return sort_importances(feature_importance)

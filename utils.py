import copy

import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from tabulate import tabulate
from sklearn.metrics import adjusted_rand_score
import tqdm
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt



def read_data(file): # return scaled data with the name of columns
    if not isinstance(file,str):
        raise TypeError('only string type is acceptable for read_data function')
    if file.endswith('.csv'):
        df = pd.read_csv(file,header=0, index_col=0)
        name_columns = list(df.columns)
        data=df.to_numpy()
        scaler = StandardScaler()
        scaler.fit(data)
        data=scaler.transform(data)
        return data,name_columns

    elif file.endswith('.npy'):
        data= np.load(file)
        scaler = StandardScaler()
        name_columns=list(range(data.shape[1]))
        scaler.fit(data)
        data=scaler.transform(data)
        return data,name_columns

    else:
        raise Exception('this file format is not supported')

def sort_importances(feature_importance):
    if not isinstance(feature_importance,dict):
        raise TypeError('only dictionary type is acceptable for sort_importances function')

    return {k: round(v,4) for k, v in sorted(feature_importance.items(), key=lambda item: abs(item[1]),reverse=True)}

def print_feature_importance(feature_importance):
    if not isinstance(feature_importance,dict):
        raise TypeError('only dictionary type is acceptable for print_feature_importance function')

    print(tabulate(feature_importance.items(),headers=['feature name','importance']))

def show_feature_importance(feature_importance):
    pass

def comparison_different_methods(data,list_of_FIs,list_of_labels,name_columns,clustering_algorithm=KMeans(n_clusters=2)):
    main_clustering_algorithm = copy.deepcopy(clustering_algorithm)
    clustering_algorithm = copy.deepcopy(main_clustering_algorithm)
    lbl = clustering_algorithm.fit_predict(data)
    list_of_results = []
    for x in list_of_FIs:
        list_of_results.append(list())
    for i in tqdm.tqdm(range(1,len(name_columns)+1)):
        for j in range(len(list_of_FIs)):
            selected=[]
            for x in list(list_of_FIs[j].keys())[:i]:
                selected.append(name_columns.index(x))
            clustering_algorithm = copy.deepcopy(main_clustering_algorithm)
            t_lbl = clustering_algorithm.fit_predict(data[:,selected])
            list_of_results[j].append(adjusted_rand_score(t_lbl,lbl))

    for i in range(len(list_of_labels)):
        plt.plot(list_of_results[i],label=list_of_labels[i])

    plt.legend()
    plt.show()


from model import _standardize , euclidean_distance
def wcss_fun(df ,model):
    all_sum = 0
    for i in range(model.K):
        distances =euclidean_distance( _standardize(df.iloc[model.clusters[i]] , model.mean , model.std) , model.centroids[i]) **2
        all_sum  += sum(distances)
    return all_sum
def a(df , model , idx):
    label = int(model.labels[idx])
    distances = euclidean_distance( _standardize(df.iloc[idx] , model.mean , model.std) , _standardize(df.iloc[model.clusters[label]] , model.mean , model.std))
    try:
        return sum(distances ) / (len(model.clusters[label]) - 1)
    except ZeroDivisionError:
        return 0
def b(df , model , idx):
    label = int(model.labels[idx])
    distances = []
    for i in range(model.K):
        if i != label:
            dist = euclidean_distance( _standardize(df.iloc[idx] , model.mean , model.std) , _standardize(df.iloc[model.clusters[i]] , model.mean , model.std))
            distances.append( sum(dist) / len(model.clusters[i]) )
    return min(distances)
def s(df , model , idx):
    a_val = a(df , model , idx)
    b_val = b(df , model , idx)
    return (b_val - a_val) / max(a_val , b_val)
def silhouette_score(df , model):
    length = len(df)
    scores = [s(df , model ,idx) for idx in range(length)]
    return sum(scores) / length
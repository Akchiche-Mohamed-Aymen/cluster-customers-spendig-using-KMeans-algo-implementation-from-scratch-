from model import KMeans  , _standardize , euclidean_distance
import pandas as pd

def wcss_fun(model):
    all_sum = 0
    for i in range(model.K):
        distances =euclidean_distance( _standardize(df.iloc[model.clusters[i]] , model.mean , model.std) , model.centroids[i]) **2
        all_sum  += sum(distances)
    return all_sum
df = pd.read_csv('customers.csv').drop(columns=["Region"])
wcss = []
try:
    for i in range(2 , 11):
        model = KMeans( K = i )
        model.fit(df)
        wcss.append(wcss_fun(model))
    import matplotlib.pyplot as plt

    # Example data
    k_values = [2, 3, 4, 5, 6, 7, 8, 9, 10]

    # Plot
    plt.plot(k_values, wcss, color='blue', linestyle=':', linewidth=2, label='WCSS')
    plt.scatter(k_values, wcss, color='red', s=60, label='Points')

    # Labels and title
    plt.title('Elbow Method for Optimal K')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
    plt.legend()
    plt.grid(True)

    # Show plot
    plt.show()
except Exception as ex:
    print(ex)

""""py test.py"""
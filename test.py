from model import KMeans  
import pandas as pd
import matplotlib.pyplot as plt
from utils import silhouette_score

df = pd.read_csv('customers.csv').drop(columns=["Region"])
'''
wcss = []
try:
    for i in range(2 , 11):
        model = KMeans( K = i )
        model.fit(df)
        wcss.append(wcss_fun(model))

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
'''

model = KMeans( K = 6 )
model.fit(df)
score = silhouette_score(df , model)
print(f"k = 6 ==> Silhouette Score : " , score)
""""py test.py"""
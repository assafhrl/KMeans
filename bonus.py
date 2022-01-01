from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

MIN_K = 1
MAX_K = 10
OUTPUT_FILENAME = "elbow.png"

def df_to_list_of_lists(df):
    l = []
    for i in range(len(df)):
        sl = [float(df.iloc[i][j]) for j in df.columns]
        l.append(sl)
    return l

def calc_distance_square(point1, point2):
    return np.sum((point1-point2)**2)

def calc_inertia(data_points, clustering, centroids):
    distance_to_centroid = np.zeros(len(data_points))
    for i in range(len(data_points)):
        distance_to_centroid = calc_distance_square(data_points[i], centroids[clustering[i]])

    return np.sum(distance_to_centroid)


def main():
    iris = load_iris()
    df = pd.DataFrame(data= np.c_[iris['data'], iris['target']], columns= iris['feature_names'] + ['target'])
    data_points = df_to_list_of_lists(df)
    ks = np.arange(MIN_K, MAX_K+1)
    inertias = np.zeros(MAX_K)

    for k in ks:
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=0).fit(data_points)
        inertias[k-MIN_K] = kmeans.inertia_/len(data_points)

    fig, ax = plt.subplots()
    ax.plot(ks, inertias, linestyle='-', color='fuchsia')
    ax.annotate('elbow point', xy=(3.2, 0.8), xytext=(5,3), arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.1"))
    t = np.linspace(0, 2*np.pi, 1000)
    r = 0.25
    x = r*np.cos(t)
    y = r*np.sin(t)
    plt.plot(x + ks[2], y + inertias[2], linestyle='--', color='black')
    plt.xlabel("k")
    plt.ylabel("Average Dispersion")
    plt.title("Elbow Method to Select Optimal K Clusters")
    plt.savefig(OUTPUT_FILENAME)

if __name__ == '__main__':
    main()
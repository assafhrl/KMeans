from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

MIN_K = 1
MAX_K = 10
OUTPUT_FILENAME = "elbow.png"

def plot_graph(ks, inertias):
    # creates the circle around the elbow point
    t = np.linspace(0, 2*np.pi, 1000)
    r = 0.25
    x = r*np.cos(t)
    y = r*np.sin(t)

    # plots the graph, circle and arrow that annotates the elbow point
    fig, ax = plt.subplots()
    ax.plot(ks, inertias, linestyle='-', color='fuchsia')
    ax.annotate('elbow point', xy=(3.2, 0.8), xytext=(5,3), arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.1"))
    plt.plot(x + ks[2], y + inertias[2], linestyle='--', color='black')
    plt.xlabel("k")
    plt.ylabel("Average Dispersion")
    plt.title("Elbow Method to Select Optimal K Clusters")
    plt.savefig(OUTPUT_FILENAME)

def main():
    # loads data and cast it to numpy array
    iris = load_iris()
    df = pd.DataFrame(data= np.c_[iris['data'], iris['target']], columns= iris['feature_names'] + ['target'])
    data = df.to_numpy()

    ks = np.arange(MIN_K, MAX_K+1) # numpy array with the values of k to check
    inertias = np.zeros(MAX_K) # numpy array to save the inertia of each k

    for k in ks:
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=0).fit(data)
        inertias[k-MIN_K] = kmeans.inertia_/len(data)

    plot_graph(ks, inertias)

if __name__ == '__main__':
    main()
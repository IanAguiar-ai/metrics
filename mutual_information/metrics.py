import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_samples, silhouette_score
from sklearn.preprocessing import MinMaxScaler
from normalized_mutual_information import mutual_information

def adjust_matrix(clusters:list, dataframe = None):
    """
    Takes the array that the user passes and leaves it in the form that the code understands.

    :param clusters: Matrix with the various clusterings, the sorting can be either by row or by column
    :type clusters: list of lists or dataframe (pandas) or array (numpy)

    :param dataframe: Dataframe with numerical variables that were used for clustering.
    :type dataframe: dataframe (pandas) or array (numpy)

    ▬ return:
    • Dictionary where the first level key is the number of clusters and the second level key is the clustering;
    """

##    if type(clusters) == list and type(clusters[0]) == list:
##        d = {}
##        if not len(clusters) == len(dataframe):
##            #transpor o clusters
##            pass
##
##        for i in range(len(clusters)):
##            d[max(clusters[i])] = {"melhor_caso":clusters[i]}

    if type(clusters) == type(pd.DataFrame()):
        clusters = clusters.values.tolist()

    if type(dataframe) == type(dataframe.DataFrame()):
        dataframe = dataframe.values.tolist()

    return clusters, dataframe

def print_metrics(clusters:list, dataframe:list, clusters_real:list = None):
    c = []
    for cluster in clusters:
        c.append(max(cluster) + 1 - min(cluster))   
    calinski_harabasz_graphic(l, dataframe, c)
    davies_boulding_graphic(l, dataframe, c)
    mutual_information_graphic(l, clusters_real, c)
    Silhouette_analysis(l, dataframe, c)

def print_graph(results:dict, cr:int = -2, title:str = "title"):
    ch = list(results.values())
    colors = []
    for val in ch:
        if cr < 0:
            if val >= sorted(ch)[cr]:
                colors.append('red')
            else:
                colors.append('grey')
        else:
            if val <= sorted(ch)[cr]:
                colors.append('red')
            else:
                colors.append('grey')

    plt.bar(list(map(str,list(results.keys()))), ch, color=colors)
    plt.xlabel("Number of clusters")
    plt.ylabel(f"{title} Index")
    plt.suptitle(f"{title} by Number of Clusters")
    plt.ylim((min(ch)/max(ch)) * min(ch), max(ch) * 1.1)

    max_index = np.argmax(ch)
    min_index = np.argmin(ch)
    plt.text(max_index, ch[max_index], f'{ch[max_index]:.2f}', ha='center', va='bottom')
    plt.text(min_index, ch[min_index], f'{ch[min_index]:.2f}', ha='center', va='bottom')

    plt.show()

def calinski_harabasz_graphic(clusters:list, X = None, groups = None, title = "Calinski-Harabasz", cr = -2):
    results = {}
    
    i = 0
    for cluster in clusters:
        ch_index = calinski_harabasz_score(X, list(map(int, cluster)))
        results.update({groups[i]: ch_index})
        i += 1

    print_graph(results, title = title, cr = cr)

def davies_boulding_graphic(clusters:list, X = None, groups = None, title = "Davies-Boulding", cr = 1):
    results = {}
    
    i = 0
    for cluster in clusters:
        ch_index = davies_bouldin_score(X, list(map(int, cluster)))
        results.update({groups[i]: ch_index})
        i += 1

    print_graph(results, title = title, cr = cr)

def mutual_information_graphic(clusters:list, y:list = None,  groups = None, title = "Normalized Mutual Information", cr = -2):
    #(x:list, y:list = None, base:int = 2, **args)
    results = {}
    
    i = 0
    for cluster in clusters:
        _, ch_index = mutual_information(list(map(int, cluster)), y, print = False)
        results.update({groups[i]: ch_index})
        i += 1

    print_graph(results, title = title, cr = cr)

def Silhouette_analysis(clusters:list, X = None, groups = None, title = "Silhouette Analysis", cr = -2):
    sl_av = {}

    for n_clusters in clusters:
        cluster_labels = list(map(int, n_clusters))
        n_clusters = max(list(map(int, n_clusters)))
        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        #clusterer = KMeans(n_clusters = n_clusters, random_state=3000)
        #cluster_labels = clusterer.fit_predict(X)


        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(X, cluster_labels)
        #silhouette_avg_2 = silhouette_score(X, list(map(int, cord_code_[n_clusters]["melhor_caso"])))
        #print("For n_clusters =", n_clusters + 1, "The average silhouette_score is :", silhouette_avg)

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(X, cluster_labels)

        y_lower = 10
        for i in range(n_clusters + 1):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = []
            for j in range(len(cluster_labels)):
                if i == cluster_labels[j]:
                    ith_cluster_silhouette_values.append(sample_silhouette_values[j])
            #ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = len(ith_cluster_silhouette_values)#.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / (n_clusters + 1))
            ax1.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                facecolor = color,
                edgecolor = color,
                alpha = 0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
        sl_av[n_clusters] = silhouette_avg

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        #colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        resp = []
        for i in range(len(cluster_labels)):
            resp.append(cluster_labels[i] / n_clusters)
        colors = cm.nipy_spectral(resp)
        ax2.scatter(
            X[:, 0], X[:, 1], marker=".", s=100, lw=0, alpha=1, c=colors, edgecolor="k"
        )

        # Labeling the clusters
##        centers = clusterer.cluster_centers_
##        # Draw white circles at cluster centers
##        ax2.scatter(
##            centers[:, 0],
##            centers[:, 1],
##            marker="o",
##            c="white",
##            alpha=1,
##            s=200,
##            edgecolor="k",
##        )

##        for i, c in enumerate(centers):
##            ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")

        plt.suptitle(
            f"Silhouette analysis for K-Means clustering on sample data with n_clusters = {n_clusters}, Value of silhouette = {str(silhouette_avg)[:6]}",
            fontsize=14,
            fontweight="bold",
        )

        plt.show()

    print_graph(sl_av, cr = -2, title = "Silhouette Analysis")


if __name__ == "__main__":
    #-------------------------------------------------------------------#
    #                          Obtain .JSON                             #
    #-------------------------------------------------------------------#
    import json
    import os
    from random import random

    scaler = MinMaxScaler()
    data_by_code = pd.read_csv("C:/Users/Usuario/Desktop/melhor_caso_code/DATA_CETESB_BY_CODE.csv") # Dados

    # Selecionar as colunas que você deseja normalizar
    X = data_by_code.iloc[1:, 3:].values

    # Normalizar as colunas selecionadas
    X = scaler.fit_transform(X)

    y_simulado = [int(random() * 7) for i in range(len(X))]
    
    c = [2,3,6,9,12]
    l = []
    for i in c:
        cord = json.load(open(f"C:/Users/Usuario/Desktop/melhor_caso_code/melhor_caso_{i}.json", "r"))

        for i in range(len(cord["lat"]) - 1, 0, -1):
          if cord["long"][i] < 30:
            cord["long"].pop(i)
            cord["lat"].pop(i)
            cord["melhor_caso"].pop(i)

        for i in range(len(cord["lat"])):
          cord["long"][i] *= -1
          cord["lat"][i] *= -1

        l.append(cord["melhor_caso"])


    #Tenho a matriz...
    
    #calinski_harabasz_graphic(l, X, c)
    #davies_boulding_graphic(l, X, c)
    #mutual_information_graphic(l, y_simulado, c)
    #Silhouette_analysis(l, X, c)
    print_metrics(l,X ,y_simulado)

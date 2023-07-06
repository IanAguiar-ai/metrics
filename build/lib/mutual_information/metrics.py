import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_samples, silhouette_score
from sklearn.preprocessing import MinMaxScaler

def count(x, y, v, v_):
    c = 0
    for i in range(len(x)):
        if x[i] == v and y[i] == v_:
            c += 1
    return c

def marginals(m:list):
    """
    Calculates the margins of a matrix.

    :param m: Matrix
    :type m: list of lists

    ▬ return:
    • float or int: marginal of x;
    • float or int: marginal of y;
    • float or int: total sum of marginal y
    """
    marg_y = [sum(m[i]) for i in range(len(m))]

    marg_x = []
    for i in range(len(m[0])):
        s = 0
        for j in range(len(m)):
            s += m[j][i]
        marg_x.append(s)

    total = sum(marg_y)

    if sum(marg_y) != sum(marg_x):
        print("As marginais estão erradas!")
        print("X",marg_x)
        print("Y",marg_y)

    return marg_x, marg_y, total

def entropi(x:list, t:int, base:int = 2):
    """
    Calculates the entropy of a vector.

    :param x: Sampling vector
    :type x: list

    :param t: Total matrix samples
    :type t: int

    :param base: Base of logarithm, a priori is 2
    :type base: int

    ▬ return:
    • float: entropy value
    """
    from math import log
    resp = 0
    for i in x:
        try:
            resp += i/t * log(i/t, base)
        except ValueError:
            pass
    return -resp

def transform_in_numerical(k:list):
    """
    Convert strings to numbers.

    :param k: List with strings
    :type k: list
    
    ▬ return:
    • list: values that were in string now in a list with ints
    """
    K = {}
    k_ = sorted(k)
    n = 0
    for i in range(len(k)):
        if not k_[i] in K:
            K[k_[i]] = n
            n += 1
        else:
            pass
    return K

def number_of_samples(x:list, y:list, precision:float = 0.95, factor:float = 1.2, **args):
    """
    It makes an approximation by simulation of the number of samples necessary for the normalized mutual information to be reasonable.

    :param x: Sample observation list 1
    :type x: list

    :param y: Sample observation list 2
    :type y: list

    :param precision: Is the minimum number of times the simulation must have its maximum at a given alpha value, where alpha = 1 - precision
    :type precision: float

    :param factor: Demand growth rate
    :type factor: float

    Other parameters:
    :param print: Whether the result will be printed on the screen.
    :type print: bool
    
    ▬ return:
    • int: Optimum value of samples
    """
    from random import random
    from math import log
    
    X = transform_in_numerical(x)
    for i in range(len(x)):
        x[i] = X[x[i]]

    Y = transform_in_numerical(y)
    for i in range(len(y)):
        y[i] = Y[y[i]]

    class_x = max(X.values()) + 1
    class_y = max(Y.values()) + 1
        
    k_ = int(log(10, factor)) + 1
    while True:
        k = int(factor**k_)
        NMI = []
        for n in range(int(3 / (1 - precision))):
            lista1 = [int(random()*class_x) for i in range(k)]
            lista2 = [int(random()*class_y) for i in range(k)]
            _, nmi = mutual_information(lista1, lista2, print = False)
            NMI.append(nmi)

        if sum(NMI)/len(NMI) < 1 - precision and max(NMI) < 1 - precision:
            if "print" in args and args["print"] == False:
                pass
            else:
                print(f"For {class_x}-{class_y}, the ideal number of samples with {precision} precision is {k}\nMean: {sum(NMI)/len(NMI)}\nMax: {max(NMI)}")
            return k
        
        if "print" in args and args["print"] == False:
            pass
        else:
            print(f"{k} samples fail...")
        k_ += 1
    return k

def mutual_information(x:list, y:list = None, base:int = 2, **args):
    """
    Calculates mutual information
    
    :param x: Sample observation list 1
    :type x: list

    :param y: Sample observation list 2
    :type y: list

    :param base: Base of logarithm, a priori is 2
    :type base: int

    Other parameters:
    :param print: Whether the result will be printed on the screen.
    :type print: bool
    
    ▬ return:
    • float: mutual information
    • float: normalized mutual information
    """
    from math import log

    if y == None and type(x[0]) == list:
        if not "print" in args or args["print"] != False:
            print("Working with confusion matrix")
        
        marg_x, marg_y, total = marginals(x)

        resp = 0
        for i in range(len(x)):
            for j in range(len(x[i])):
                try:
                    resp += (x[i][j]/total) * log( ( (x[i][j]/total) / ( (marg_x[j]/total) * (marg_y[i]/total) ) ) , base)
                except ValueError:
                    resp += 0
                except:
                    pass
        
        ent_x = entropi(marg_x, total, base)
        ent_y = entropi(marg_y ,total, base)

        try:
            normalized = 2*resp / (ent_x + ent_y)
        except ZeroDivisionError:
            print("The sample is too small")
            normalized = 1

    else:
        if y == None:
            print("y has not been defined")
            return
            
        if type(x[0]) == str:
            if not "print" in args or args["print"] != False:
                print("Passing categorical variable X to numeric categorical...")

            X = transform_in_numerical(x)
            for i in range(len(x)):
                x[i] = X[x[i]]  
        
            
        if type(y[0]) == str:
            if not "print" in args or args["print"] != False:
                print("Passing categorical variable Y to numeric categorical...")
                
            Y = transform_in_numerical(y)
            for i in range(len(y)):
                y[i] = Y[y[i]]
            
        m = [[0 for j in range(int(max(x)) + 1)] for i in range(int(max(y)) + 1)]

        if len(x) != len(y):
            print(f"Inputs must be the same size\nINPUT[1] = {len(x)}\nINPUT[2] = {len(y)}")
            return

        for i in range(len(m)):
            for j in range(len(m[0])):
                m[i][j] = count(x, y, j, i)

        if not "print" in args or args["print"] != False:
            i = 0
            for v in m:
                if "Y" in locals():
                    print(v, "--", list(Y.keys())[i])
                else:
                    print(v, "--", i)
                i += 1

            print(str(["|" for i in range(max(x) + 1)]).replace("["," ").replace("]","").replace("'","").replace(","," "))
            if "X" in locals():
                print(str([i for i in X.keys()]).replace("["," ").replace("]","").replace("'","").replace(","," "))
            else:
                print(str([i for i in range(max(x) + 1)]).replace("["," ").replace("]","").replace("'","").replace(","," "))

        marg_x, marg_y, total = marginals(m)

        if sum(marg_y) != sum(marg_x):
            print("Margins are wrong")
            print("X", marg_x)
            print("Y", marg_y)

        resp = 0
        for i in range(len(m)):
            for j in range(len(m[i])):
                try:
                    resp += (m[i][j]/total) * log( ( (m[i][j]/total) / ( (marg_x[j]/total) * (marg_y[i]/total) ) ) , base)
                except ValueError:
                    resp += 0
                except:
                    pass

        ent_x = entropi(marg_x, total, base)
        ent_y = entropi(marg_y ,total, base)
        try:
            normalized = 2*resp / (ent_x + ent_y)
        except ZeroDivisionError:
            print("The sample is too small")
            normalized = 1

    if not "print" in args or args["print"] != False:
        print(f"\nCertainty knowing the result of X knowing Y: {1 / base ** (ent_y - resp)} ~ 1 in {base ** (ent_y - resp)}")
        print(f"Certainty knowing the result of Y knowing X: {1 / base ** (ent_x - resp)} ~ 1 in {base ** (ent_x - resp)}")
        
        print(f"\nEntropy in logarithm base {base}:\nX: {str(ent_x)} -> {base}**X = {str(base**ent_x)}\nY: {str(ent_y)} -> {base}**Y = {str(base**ent_y)}")
        print(f"Mutual Information in base {base}: {str(resp)}\nNormalized Mutual Information: {str(normalized)}\n" +  "-" * 50 + "\n")

        if normalized < 0.05:
            if number_of_samples(x, y, 0.95, 2, print = False) < len(x):
                print("Strong evidence that the distribution is random\n")
        elif normalized > 0.95:
            if number_of_samples(x, y, 0.95, 2, print = False) < len(x):
                print("Strong evidence that the distribution is not random\n")
        
    return resp, normalized

def print_metrics(clusters:list, dataframe:list, clusters_real:list = None, **args):
    """
    Assemble some graphs of metrics, the metrics are:
    • silhouette analysis;
    • calinski_harabasz;
    • davies_boulding;
    • mutual_information;

    :param clusters: clustering of some method
    :type clusters: list of lists, pandas dataframe or numpy array

    :param dataframe: The dataframe that was used for clustering
    :type dataframe: dataframe od pandas or list of lists

    :param clusters_real: expected response, this parameter is optional, it will be used to print normalized mutual information
    :type clusters_real: list

    ▬ return:
    • None
    """
    
    if type(clusters) == type(pd.DataFrame()):
        clusters = clusters.T.values.tolist()

    if type(dataframe) == type(pd.DataFrame()):
        dataframe = dataframe.values    
    
    c = []
    for cluster in clusters:
        c.append(max(cluster) + 1 - min(cluster))

    r1 = silhouette_analysis(clusters, dataframe, c, print = False)
    r2 = calinski_harabasz(clusters, dataframe, c, print = False)
    r3 = davies_boulding(clusters, dataframe, c, print = False)
    if clusters_real != None:
        r4 = mutual_information_graph(clusters, clusters_real, c, print = False)

    if not "figsize" in args:
        args["figsize"] = (14, 8)

    if not "color" in args:
        args["color"] = ["green", "grey"]

    if not "best" in args:
        args["best"] = 2

    fig, ax = plt.subplots(nrows = 2, ncols = 2, figsize = args["figsize"])
    try:
        variables = [[0,0,1,1],
                     [0,1,0,1],
                     [r1, r2, r3, r4],
                     [-args["best"], -args["best"], args["best"] - 1, -args["best"]],
                     ["Silhouette Analysis",
                      "Calinski-Harabasz",
                      "Davies-Boulding",
                      "Normalized Mutual Information"]]
    except:
        variables = [[0,0,1],
                     [0,1,0],
                     [r1, r2, r3],
                     [-args["best"], -args["best"], args["best"] - 1],
                     ["Silhouette Analysis",
                      "Calinski-Harabasz",
                      "Davies-Boulding"]]
    
    for i, j, ch_, cr, title in zip(*variables):
        #print(i, j, cr)
        ch = list(ch_.values())
        colors = []
        for val in ch:
            if cr < 0:
                if val >= sorted(ch)[cr]:
                    colors.append(args["color"][0])
                else:
                    colors.append(args["color"][1])
            else:
                if val <= sorted(ch)[cr]:
                    colors.append(args["color"][0])
                else:
                    colors.append(args["color"][1])

        ax[i, j].bar(list(map(str, list(map(int, list(ch_.keys()))))), ch, color=colors)
        ax[i, j].set_xlabel("Number of clusters")
        ax[i, j].set_ylabel(f"{title} Index")
        #ax[i, j].set_suptitle(f"{title} by Number of Clusters")
        ax[i, j].set_ylim((min(ch)/max(ch)) * min(ch), max(ch) * 1.1)

        max_index = np.argmax(ch)
        min_index = np.argmin(ch)
        ax[i, j].text(max_index, ch[max_index], f'{ch[max_index]:.2f}', ha='center', va='bottom')
        ax[i, j].text(min_index, ch[min_index], f'{ch[min_index]:.2f}', ha='center', va='bottom')

    if len(variables[0]) == 3:
            ax[1][1].text(0.15, 0.4, 'To plot the normalized mutual information\nyou also need to pass a list of expected\nvalues as an argument.', fontsize = 12)
            
    fig.suptitle(f"All Metrics")
    fig.show()

    if not "set_fontsize" in args:
        args["set_fontsize"] = 10

        
    if not "figsize_table" in args:
        args["figsize_table"] = (9, 3)

    if not "scale" in args:
        args["scale"] = (1, 2)

    def conv(x, decimals = 4):
        x = str(x)
        return x[:x.find(".") + decimals]

    def mod(a):
        if a < 0:
            return -a
        return a

    def best(l:list, func = max):
        l_ = list(map(float, l))
        for i in range(len(l_)):
            if mod(l_[i]) > mod(func(l_)) * 0.9 and mod(l_[i]) < mod(func(l_)) * 1.1:
                l[i] = l[i] + "*"
        return l


    if len(variables[0]) == 3:
        df = pd.DataFrame({"Number of\nClusters":r1.keys(),
                           "Silhouette\nAnalysis":best(list(map(conv, r1.values()))),
                           "Calinski\nHarabasz":best(list(map(conv, r2.values()))),
                           "Davies\nBoulding":best(list(map(conv, r3.values())), min)})

    else:
        df = pd.DataFrame({"Number of\nClusters":r1.keys(),
                           "Silhouette\nAnalysis":best(list(map(conv, r1.values()))),
                           "Calinski\nHarabasz":best(list(map(conv, r2.values()))),
                           "Davies\nBoulding":best(list(map(conv, r3.values())), min),
                           "Normalized\nMutual Information":best(list(map(conv, r4.values())))})

    fig, ax = plt.subplots(figsize = args["figsize_table"])
    ax.axis('off')

    table = ax.table(cellText = df.values, colLabels = df.columns, loc = 'center', cellLoc = 'center')

    table.auto_set_font_size(False)
    table.set_fontsize(args["set_fontsize"])
    table.scale(*args["scale"])

    fig.show()  

     

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

def calinski_harabasz(clusters:list, X = None, groups = None, title = "Calinski-Harabasz", cr = -2, print = True):
    """
    Metric

    :param clusters: clustering of some method
    :type clusters: list of lists, pandas dataframe or numpy array

    :param X: The dataframe that was used for clustering
    :type X: dataframe od pandas or list of lists

    :param groups: Group numbering, é opcional
    :type groups: list

    :param print: Whether it should be printed or not
    :type print: bool

    ▬ return:
    if print == True:
        • None
    else:
        • Results
    """
    results = {}
    
    i = 0
    for cluster in clusters:
        ch_index = calinski_harabasz_score(X, list(map(int, cluster)))
        results.update({groups[i]: ch_index})
        i += 1

    if print:
        print_graph(results, title = title, cr = cr)
    else:
        return results

def davies_boulding(clusters:list, X = None, groups = None, title = "Davies-Boulding", cr = 1, print = True):
    """
    Metric

    :param clusters: clustering of some method
    :type clusters: list of lists, pandas dataframe or numpy array

    :param X: The dataframe that was used for clustering
    :type X: dataframe od pandas or list of lists

    :param groups: Group numbering, é opcional
    :type groups: list

    :param print: Whether it should be printed or not
    :type print: bool

    ▬ return:
    if print == True:
        • None
    else:
        • Results
    """
    results = {}
    
    i = 0
    for cluster in clusters:
        ch_index = davies_bouldin_score(X, list(map(int, cluster)))
        results.update({groups[i]: ch_index})
        i += 1
        
    if print:
        print_graph(results, title = title, cr = cr)
    else:
        return results

def mutual_information_graph(clusters:list, y:list = None,  groups = None, title = "Normalized Mutual Information", cr = -2, print = True):
    """
    Metric

    :param clusters: clustering of some method
    :type clusters: list of lists, pandas dataframe or numpy array

    :param X: The dataframe that was used for clustering
    :type X: dataframe od pandas or list of lists

    :param groups: Group numbering, é opcional
    :type groups: list

    :param print: Whether it should be printed or not
    :type print: bool

    ▬ return:
    if print == True:
        • None
    else:
        • Results
    """
    #(x:list, y:list = None, base:int = 2, **args)
    results = {}
    
    i = 0
    for cluster in clusters:
        _, ch_index = mutual_information(list(map(int, cluster)), y, print = False)
        results.update({groups[i]: ch_index})
        i += 1

    if print:
        print_graph(results, title = title, cr = cr)
    else:
        return results

def silhouette_analysis(clusters:list, X = None, groups = None, title = "Silhouette Analysis", cr = -2, size = (18, 7), print = True):
    """
    Metric

    :param clusters: clustering of some method
    :type clusters: list of lists, pandas dataframe or numpy array

    :param X: The dataframe that was used for clustering
    :type X: dataframe od pandas or list of lists

    :param groups: Group numbering, é opcional
    :type groups: list

    :param print: Whether it should be printed or not
    :type print: bool

    ▬ return:
    if print == True:
        • None
    else:
        • Results
    """
    sl_av = {}

    for n_clusters in clusters:
        cluster_labels = list(map(int, n_clusters))
        n_clusters = max(list(map(int, n_clusters)))
        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(*size)

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
        sl_av[n_clusters + 1] = silhouette_avg

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
##        colors = []
##        for i in range(len(clusters)):
##            for j in range(len(clusters[0])):
##                #print(clusters[i][j])
##                colors.append(clusters[i][j]/n_clusters)
##        colors = cm.nipy_spectral(colors)
        colors = cm.nipy_spectral(np.array(cluster_labels).astype(float) / (n_clusters + 1))
        #colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
##        resp = []
##        for i in range(len(cluster_labels)):
##            resp.append(cluster_labels[i] / n_clusters)
##        colors = cm.nipy_spectral(resp)
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
            f"Silhouette analysis for K-Means clustering on sample data with n_clusters = {n_clusters + 1}, Value of silhouette = {str(silhouette_avg)[:6]}",
            fontsize=14,
            fontweight="bold",
        )

        plt.show() ###voltar depois

    if print:
        print_graph(sl_av, cr = -2, title = "Silhouette Analysis")
    else:
        return sl_av


if __name__ == "__main__":
    #-------------------------------------------------------------------#
    #                          Obtain .JSON                             #
    #-------------------------------------------------------------------#
    import json
    import os
    from random import random

    scaler = MinMaxScaler()
    try:
        data_by_code = pd.read_csv("C:/Users/Usuario/Desktop/melhor_caso_code/DATA_CETESB_BY_CODE.csv") # Dados
    except:
        data_by_code = pd.read_csv("C:/Users/Ian_Anjos/Desktop/DATA_CETESB_BY_CODE.csv") # Dados

    # Selecionar as colunas que você deseja normalizar
    X = data_by_code.iloc[1:, 3:].values

    # Normalizar as colunas selecionadas
    X = scaler.fit_transform(X)

    y_simulado = [int(random() * 7) for i in range(len(X))]
    
    c = [2,3,6,9,12]
    l = []
    for i in c:
        try:
            cord = json.load(open(f"C:/Users/Usuario/Desktop/melhor_caso_code/melhor_caso_{i}.json", "r"))
        except:
            cord = json.load(open(f"C:/Users/Ian_Anjos/Desktop/melhor_caso_code/melhor_caso_{i}.json", "r"))

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
    
##    calinski_harabasz(l, X, c)
##    davies_boulding(l, X, c)
##    mutual_information_graph(l, y_simulado, c)
##    silhouette_analysis(l, X, c)
    print_metrics(l,X ,y_simulado)
##        
##    print_metrics(pd.DataFrame({'2': [int(random()*2) for i in range(50)],
##                                '3': [int(random()*3) for i in range(50)],
##                                '6': [int(random()*6) for i in range(50)],
##                                '8': [int(random()*8) for i in range(50)]}),
##                  pd.DataFrame({'2': [int(random()*2) for i in range(50)],
##                                '3': [int(random()*3) for i in range(50)],
##                                '6': [int(random()*6) for i in range(50)],
##                                '8': [int(random()*8) for i in range(50)]}))
        

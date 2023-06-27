def adjust_matrix(clusters:list, dataframe):
    """
    Takes the array that the user passes and leaves it in the form that the code understands.

    :param clusters: Matrix with the various clusterings, the sorting can be either by row or by column
    :type clusters: list of lists or dataframe (pandas) or array (numpy)

    :param dataframe: Dataframe with numerical variables that were used for clustering.
    :type dataframe: dataframe (pandas) or array (numpy)

    ▬ return:
    • Dictionary where the first level key is the number of clusters and the second level key is the clustering;
    """

    if type(clusters) == list and type(clusters[0]) == list:
        d = {}
        if not len(clusters) == len(dataframe):
            #transpor o clusters
            pass

        for i in range(len(clusters)):
            d[max(clusters[i])] = {"melhor_caso":clusters[i]}

    elif type(clusters) == dataframenaoseioobjeto:
        pass

if __name__ == "__main__":
    #-------------------------------------------------------------------#
    #                          Obtain .JSON                             #
    #-------------------------------------------------------------------#
    import json
    import os

    c = [2,6,9,12]
    l = []
    for i in c:
        cord = json.load(open(f"./melhor_caso_{i}.json", "r"))

        for i in range(len(cord["lat"]) - 1, 0, -1):
          if cord["long"][i] < 30:
            cord["long"].pop(i)
            cord["lat"].pop(i)
            cord["melhor_caso"].pop(i)

        for i in range(len(cord["lat"])):
          cord["long"][i] *= -1
          cord["lat"][i] *= -1

        l.append(cord["melhor_caso"])


    ### Testando.


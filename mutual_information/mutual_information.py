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


if __name__ == "__main__":
##    from sklearn.metrics import mutual_info_score, normalized_mutual_info_score

    # Exemplo de duas listas de valores
##    lista1 = ["a", "b", "a", "a", "c", "c", "b"]
##    lista2 = ["ab", "ab", "aa", "ac", "ac", "aa", "ad"]

#Moeda que controla dado, se cair cara o dado cai 1 a 3

##    lista1 = [int(random()*2) for i in range(100)]
##    lista2 = []
##
##    for i in lista1:
##        if i == 0:
##            lista2.append(int(random()*3)+1)
##        else:
##            lista2.append(int(random()*6)+1)


    #Uno 4x4:
##    l = ["a","b","c","d"]
##
##    lista1 = [l[int(random()*4)] + str(int(random()*4)) for i in range(100000)]
##
##    lista2 = []
##
##    for i in range(len(lista1)):
##        if random() > 0.5:
##            lista2.append(l[int(random()*4)] + lista1[i][1])
##        else:
##            lista2.append(lista1[i][0] + str(int(random()*4)))

##    for categorias in range(2,10+1):
##        k = int(categorias ** (1.62) * 10)
##        print("-"*50)         
##        NMI = []
##        for n in range(100):
##            lista1 = [int(random()*categorias) for i in range(k)]
##            lista2 = [int(random()*categorias) for i in range(k)]
##            _, nmi = mutual_information(lista1, lista2, print = False)
##            NMI.append(nmi)
##
##        print(f"Para {categorias} categorias e {k} amostras em média {sum(NMI)/len(NMI)} e maximo de {max(NMI)}")
        

##    # Calcular a informação mútua
##    mi = mutual_info_score(lista1, lista2)
##    print("Informação Mútua:", mi)
##
##    # Calcular a informação mútua normalizada
##    nmi = normalized_mutual_info_score(lista1, lista2)
##    print("Informação Mútua Normalizada:", nmi)

##    a, b = 7, 7
##    lista1 = [int(random()*a) for i in range(100)]
##    lista2 = [int(random()*b) for i in range(100)]
##    number_of_samples(lista1, lista2, 0.95, 1.1)

##    text = """
##Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.
##It is a long established fact that a reader will be distracted by the readable content of a page when looking at its layout. The point of using Lorem Ipsum is that it has a more-or-less normal distribution of letters, as opposed to using 'Content here, content here', making it look like readable English. Many desktop publishing packages and web page editors now use Lorem Ipsum as their default model text, and a search for 'lorem ipsum' will uncover many web sites still in their infancy. Various versions have evolved over the years, sometimes by accident, sometimes on purpose (injected humour and the like).
##There are many variations of passages of Lorem Ipsum available, but the majority have suffered alteration in some form, by injected humour, or randomised words which don't look even slightly believable. If you are going to use a passage of Lorem Ipsum, you need to be sure there isn't anything embarrassing hidden in the middle of text. All the Lorem Ipsum generators on the Internet tend to repeat predefined chunks as necessary, making this the first true generator on the Internet. It uses a dictionary of over 200 Latin words, combined with a handful of model sentence structures, to generate Lorem Ipsum which looks reasonable. The generated Lorem Ipsum is therefore always free from repetition, injected humour, or non-characteristic words etc.
##Contrary to popular belief, Lorem Ipsum is not simply random text. It has roots in a piece of classical Latin literature from 45 BC, making it over 2000 years old. Richard McClintock, a Latin professor at Hampden-Sydney College in Virginia, looked up one of the more obscure Latin words, consectetur, from a Lorem Ipsum passage, and going through the cites of the word in classical literature, discovered the undoubtable source. Lorem Ipsum comes from sections 1.10.32 and 1.10.33 of "de Finibus Bonorum et Malorum" (The Extremes of Good and Evil) by Cicero, written in 45 BC. This book is a treatise on the theory of ethics, very popular during the Renaissance. The first line of Lorem Ipsum, "Lorem ipsum dolor sit amet..", comes from a line in section 1.10.32.
##""".lower()
##
##    print(len(text))
##
##    text1 = list(map(str,list(map(ord,text))))
##    text2 = list(map(str,list(map(ord,text[1:] + text[0:1]))))
##    
##    mutual_information(text1, text2)
    
##    mutual_information([[2216/2-1,1],[1,2216/2-1]])
##
##    number_of_samples(text1, text2, 0.95, 2)

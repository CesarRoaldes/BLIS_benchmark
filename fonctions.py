#Fonctions utiles 

def time_count_np(name, small=False):
    import numpy as np
    import os
    import pickle
    import time
    
    PATH = os.path.abspath('')
    np.__config__.show()
    
    if not small:
        # Dictionnaire collectant les resultats
        res = {}

        # Nombre de boucle pour chaque mesure
        B = 5

        print("### Temps d'execution moyen d'un matrice de taille :")
        for N in [50 * x for x in np.arange(1, 101)]:
            a = np.random.random((N, N))
            b = np.random.random((N, N))
            tot = 0.0
            for i in np.arange(B):
                start = time.time()
                np.matmul(a, b)
                tot += time.time() - start
            res[N] = tot / B
            print("\t- ({}x{})\t:\t\t{}\ts.".format(N, N, res[N]))
        results=res
        pickle.dump(results, open(PATH+"/data/"+name+".pkl", "wb"))
        return results

    else:
        # Dictionnaire collectant les resultats
        res = {}

        # Nombre de boucle pour chaque mesure
        B = 5

        print("### Temps d'execution moyen d'un matrice de taille :")
        for N in [50 * x for x in np.arange(1, 51)]:
            a = np.random.random((N, N))
            b = np.random.random((N, N))
            tot = 0.0
            for i in np.arange(B):
                start = time.time()
                np.matmul(a, b)
                tot += time.time() - start
            res[N] = tot / B
            print("\t- ({}x{})\t:\t\t{}\ts.".format(N, N, res[N]))
        results=res
        pickle.dump(results, open(PATH+"/data/"+name+"_SMALL.pkl", "wb"))
        return results

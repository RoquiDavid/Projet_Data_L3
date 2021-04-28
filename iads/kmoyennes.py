# ---------------------------
# Fonctions pour les k-moyennes

# Importations nécessaires pour l'ensemble des fonctions de ce fichier:
import pandas as pd
import matplotlib.pyplot as plt

import math
import random

import numpy as np

def normalisation(A):
    return (A - A.min())/(A.max() - A.min())


# Fonctions distances
def dist_vect(V1,V2):
    return np.linalg.norm(V1 - V2)

# Calculs de centroïdes :
def centroide(E):
    return np.median(E, axis=0)

# Inertie des clusters :
def inertie_cluster(norm):
    center = centroide(norm)
    inertie = 0
    for v in norm:
        inertie = inertie + dist_vect(v,center)**2
    return inertie


# -------
# Algorithmes des K-means :

def initialisation(k, norm):
    r = list(range(norm.shape[0]))
    random.shuffle(r)
    return norm[r[0:3]]


def plus_proche(exemple,norm):   
    dist_min = float("inf")
    index_centre = 0
    index = 0
    for i in range (len(norm)):
        dist = dist_vect(exemple, norm[i])
        if dist < dist_min:
            dist_min = dist
            index_centre = index
        index += 1
    return index_centre


def affecte_cluster(norm, mat_aff):
    dic = {}
    for i in range(len(norm)):
        index_centre = plus_proche(norm[i],mat_aff)
        if index_centre in dic:
            dic[index_centre].append(i)
        else:
            dic[index_centre] = [i]
    return dic


def nouveaux_centroides(norm, mat_aff):
    df_learn = pd.DataFrame(norm)
    df_res = pd.DataFrame()
    for centre in mat_aff:
        n_centre = df_learn.iloc[mat_aff[centre]]
        n_centre = n_centre.mean()
        df_res = df_res.append(n_centre, ignore_index = True)
    
    df_res = df_res.to_numpy()
    return df_res

def inertie_globale(norm, mat_aff):
    inertie = 0
    for centre in mat_aff:
        cluster = norm[mat_aff[centre]]
        inertie += inertie_cluster(cluster)
    return inertie


def kmoyennes(k, norm, epsilon, iter_max, verbose = False):
    centroides = initialisation(k, norm)
    mat_aff = affecte_cluster(norm, centroides)
    centroides = nouveaux_centroides(norm, mat_aff)
    inertie = inertie_globale(norm, mat_aff)
    
    for iter in range(iter_max):
        mat_aff = affecte_cluster(norm, centroides)
        centroides = nouveaux_centroides(norm, mat_aff)
        n_inertie = inertie_globale(norm, mat_aff)
        if verbose == True:
            print("iteration ", iter, " Inertie : ", n_inertie, " Difference: ",  abs(n_inertie - inertie))
        if abs(n_inertie - inertie) < epsilon:
            break
        else:
            inertie = n_inertie 
    return centroides, mat_aff

def affiche_resultat(norm, lCentres, mat_aff):
    for i, centre in enumerate(mat_aff):
        res = norm[mat_aff[centre]]
        plt.scatter(res[:,0],res[:,1], color=("C" + str(i % 1000)))
    plt.scatter(lCentres[:,0],lCentres[:,1],color='r',marker='x')


def sep_clusters(norm):
    d_min = float("inf")
    for i in range (len(norm)):
        for j in range (len(norm)):
            if i != j:
                if dist_vect(norm[i], norm[j]) < d_min:
                    d_min = dist_vect(norm[i], norm[j])
    return d_min



def evaluation(norm, lCentres, mat_aff):
    return inertie_globale(norm, mat_aff)/sep_clusters(lCentres)
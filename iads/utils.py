# -*- coding: utf-8 -*-

"""
Package: iads
Fichier: utils.py
Année: semestre 2 - 2019-2020, Sorbonne Université
"""

# ---------------------------
# -*- coding: utf-8 -*-

"""
Package: iads
File: utils.py
Année: LU3IN026 - semestre 2 - 2020-2021, Sorbonne Université
"""


# Fonctions utiles pour les TDTME de LU3IN026
# Version de départ : Février 2021

# import externe
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import ticker
from matplotlib import cm

# ------------------------ 
def plot2DSet(desc,label):    
    """ ndarray * ndarray -> affichage
        la fonction doit utiliser la couleur 'red' pour la classe -1 et 'blue' pour la +1
    """
   #TODO: A Compléter    
    data_negatifs = desc[label == -1]
    #print(desc[label == -1])
    data_positifs = desc[label == +1]
    plt.scatter(data_negatifs[:,0],data_negatifs[:,1],marker='o', color='red') # 'o' pour la classe -1
    plt.scatter(data_positifs[:,0],data_positifs[:,1],marker='x', color='blue') # 'x' pour la classe +1 
    plt.grid(True)

    
def plot_frontiere(desc_set, label_set, classifier, step=30):
    """ desc_set * label_set * Classifier * int -> NoneType
        Remarque: le 4e argument est optionnel et donne la "résolution" du tracé
        affiche la frontière de décision associée au classifieur
    """
    mmax=desc_set.max(0)
    mmin=desc_set.min(0)
    x1grid,x2grid=np.meshgrid(np.linspace(mmin[0],mmax[0],step),np.linspace(mmin[1],mmax[1],step))
    grid=np.hstack((x1grid.reshape(x1grid.size,1),x2grid.reshape(x2grid.size,1)))
    
    res=np.array([ classifier.predict(grid[i,:]) for i in range(len(grid)) ])
    res=res.reshape(x1grid.shape)
    # tracer des frontieres
    # colors[0] est la couleur des -1 et colors[1] est la couleur des +1
    plt.contourf(x1grid,x2grid,res,colors=["darksalmon","skyblue"],levels=[-1000,0,1000])    
# ------------------------ 
def genere_dataset_uniform(p, n, binf=-1, bsup=1):
    
    """ int * int * float^2 -> tuple[ndarray, ndarray]
        Hyp: n est pair
        p: nombre de dimensions de la description
        n: nombre d'exemples
        les valeurs générées uniformément sont dans [binf,bsup]
    """
    #TODO: A Compléter    
    val = np.random.uniform(binf,bsup,(n,p))
    labels = np.asarray([-1 for i in range(0,n//2)] + [+1 for i in range(0,n//2)])
    return (val,labels)
    
def genere_dataset_gaussian(positive_center, positive_sigma, negative_center, negative_sigma, nb_points):
    """ les valeurs générées suivent une loi normale
        rend un tuple (data_desc, data_labels)
    """
    #TODO: A Compléter    
    pos = np.random.multivariate_normal(positive_center, positive_sigma, size=nb_points)
    neg = np.random.multivariate_normal(negative_center, negative_sigma, size=nb_points)

    desc = np.concatenate((neg,pos), axis =0)
    labels = np.asarray([-1 for i in range(0,nb_points)] + [+1 for i in range(0,nb_points)])
   
    
    return desc,labels


def crossval_strat(X, Y, n_iterations, iteration):
    # A COMPLETER
    index = np.random.permutation(len(X)) # mélange des index
    #Split des index en deux (partie test et partie app)
    index = np.array_split(index,n_iterations)
    Xt = np.array_split(X[index[0]], n_iterations)
    Yt = np.array_split(Y[index[0]], n_iterations)
    Xtest = Xt[iteration]
    Ytest = Yt[iteration]
    #On selectionne les index non utilisé pas test
    train_index = [value for tabs in index[1:] for value in tabs]

    Xa = np.array_split(X[train_index], n_iterations)
    
    Ya = np.array_split(Y[train_index], n_iterations)
    Xapp = Xa[iteration]
    Yapp = Ya[iteration]
    return Xapp, Yapp, Xtest, Ytest

# def plot2DSetMulticlass(Xm,Ym):

#     # data_negatifs = Xm[Ym == -1]
#     # data_positifs = Xm[Ym == +1]
#     # plt.scatter(data_negatifs[:,0],data_negatifs[:,1],marker='o', color='red') # 'o' pour la classe -1
#     # plt.scatter(data_positifs[:,0],data_positifs[:,1],marker='x', color='blue') # 'x' pour la classe +1 
#     # plt.grid(True)
#     label = np.unique(Ym)
#     colors = iter(cm.rainbow(np.linspace(0, 1, len(label))))
#     for classe in label:
#         data_in_classe = Xm[Ym == classe]
#         #plt.scatter(data_out_classe[:,0],data_out_classe[:,1],marker='o', color='red') # 'o' pour la classe -1
#         plt.scatter(data_in_classe[:,0],data_in_classe[:,1],marker='x', color=next(colors)) # 'x' pour la classe +1 
#         plt.grid(True)

def plot_frontiere_V3(desc_set, label_set, w, kernel, step=30, forme=1, fname="out/tmp.pdf"):
    """ desc_set * label_set * array * function * int * int * str -> NoneType
        Note: le classifieur linéaire est donné sous la forme d'un vecteur de poids pour plus de flexibilité
    """
    # ETAPE 1: construction d'une grille de points sur tout l'espace défini par les points du jeu de données
    mmax=desc_set.max(0)
    mmin=desc_set.min(0)
    x1grid,x2grid=np.meshgrid(np.linspace(mmin[0],mmax[0],step),np.linspace(mmin[1],mmax[1],step))
    grid=np.hstack((x1grid.reshape(x1grid.size,1),x2grid.reshape(x2grid.size,1)))
    #
    # Si vous avez du mal à saisir le concept de la grille, décommentez ci-dessous
    # plt.figure()
    # plt.scatter(grid[:,0],grid[:,1])
    # if True:
    #    return
    #
    # ETAPE 2: calcul de la prediction pour chaque point de la grille
    res=np.array([kernel(grid[i,:])@w for i in range(len(grid)) ])
    # pour les affichages avancés, chaque dimension est présentée sous la forme d'une matrice
    res=res.reshape(x1grid.shape) 

def k_id(x): # fonction identité (juste pour être compatible avec les kernels ensuite)
    return x

############### FOR DECISION TREE ###################

def shannon(P):
    """ list[Number] -> float
        Hypothèse: la somme des nombres de P vaut 1
        P correspond à une distribution de probabilité
        rend la valeur de l'entropie de Shannon correspondante
    """
    k = len(P)
    if k <= 1:
        return 0.0
    
    return  -sum([i * math.log(i, k) if i > 0 else 0 for i in P])
    
def entropie(labels):
    labels = np.asarray(labels)
    
    valeurs, nb_fois = np.unique(labels ,return_counts=True)
    k = len(labels)
    
    return shannon([count_label / k for count_label in nb_fois])

def classe_majoritaire(Y):
    """ Y : (array) : array de labels
        rend la classe majoritaire ()
    """
    valeurs, nb_fois = np.unique(Y,return_counts=True)
    
    return valeurs[np.argmax(nb_fois)]

class NoeudCategoriel:
    """ Classe pour représenter des noeuds d'un arbre de décision
    """
    def __init__(self, num_att=-1, nom=''):
        """ Constructeur: il prend en argument
            - num_att (int) : le numéro de l'attribut auquel il se rapporte: de 0 à ...
              si le noeud se rapporte à la classe, le numéro est -1, on n'a pas besoin
              de le préciser
            - nom (str) : une chaîne de caractères donnant le nom de l'attribut si
              il est connu (sinon, on ne met rien et le nom sera donné de façon 
              générique: "att_Numéro")
        """
        self.attribut = num_att    # numéro de l'attribut
        if (nom == ''):            # son nom si connu
            self.nom_attribut = 'att_'+str(num_att)
        else:
            self.nom_attribut = nom 
        self.Les_fils = None       # aucun fils à la création, ils seront ajoutés
        self.classe   = None       # valeur de la classe si c'est une feuille
        
    def est_feuille(self):
        """ rend True si l'arbre est une feuille 
            c'est une feuille s'il n'a aucun fils
        """
        return self.Les_fils == None
    
    def ajoute_fils(self, valeur, Fils):
        """ valeur : valeur de l'attribut de ce noeud qui doit être associée à Fils
                     le type de cette valeur dépend de la base
            Fils (NoeudCategoriel) : un nouveau fils pour ce noeud
            Les fils sont stockés sous la forme d'un dictionnaire:
            Dictionnaire {valeur_attribut : NoeudCategoriel}
        """
        if self.Les_fils == None:
            self.Les_fils = dict()
        self.Les_fils[valeur] = Fils
        # Rem: attention, on ne fait aucun contrôle, la nouvelle association peut
        # écraser une association existante.
    
    def ajoute_feuille(self,classe):
        """ classe: valeur de la classe
            Ce noeud devient un noeud feuille
        """
        self.classe    = classe
        self.Les_fils  = None   # normalement, pas obligatoire ici, c'est pour être sûr
        
    def classifie(self, exemple):
        """ exemple : numpy.array
            rend la classe de l'exemple (pour nous, soit +1, soit -1 en général)
            on rend la valeur 0 si l'exemple ne peut pas être classé (cf. les questions
            posées en fin de ce notebook)
        """
        if self.est_feuille():
            return self.classe
        if exemple[self.attribut] in self.Les_fils:
            # descente récursive dans le noeud associé à la valeur de l'attribut
            # pour cet exemple:
            return self.Les_fils[exemple[self.attribut]].classifie(exemple)
        else:
            # Cas particulier : on ne trouve pas la valeur de l'exemple dans la liste des
            # fils du noeud... Voir la fin de ce notebook pour essayer de résoudre ce mystère...
            print('\t*** Warning: attribut ',self.nom_attribut,' -> Valeur inconnue: ',exemple[self.attribut])
            return 0
    
    def to_graph(self, g, prefixe='A'):
        """ construit une représentation de l'arbre pour pouvoir l'afficher graphiquement
            Cette fonction ne nous intéressera pas plus que ça, elle ne sera donc pas expliquée            
        """
        if self.est_feuille():
            g.node(prefixe,str(self.classe),shape='box')
        else:
            g.node(prefixe, self.nom_attribut)
            i =0
            for (valeur, sous_arbre) in self.Les_fils.items():
                sous_arbre.to_graph(g,prefixe+str(i))
                g.edge(prefixe,prefixe+str(i), valeur)
                i = i+1        
        return g
    
    
def construit_AD(X,Y,epsilon,LNoms = []):
    """ X,Y : dataset
        epsilon : seuil d'entropie pour le critère d'arrêt 
        LNoms : liste des noms de features (colonnes) de description 
    """
    
    entropie_ens = entropie(Y)
    if (entropie_ens <= epsilon):
        # ARRET : on crée une feuille
        noeud = NoeudCategoriel(-1,"Label")
        noeud.ajoute_feuille(classe_majoritaire(Y))
    else:
        min_entropie = 1.1
        i_best = -1
        Xbest_valeurs = None
        
        ############################# DEBUT ########
        
        # COMPLETER CETTE PARTIE : ELLE DOIT PERMETTRE D'OBTENIR DANS
        # i_best : le numéro de l'attribut qui minimise l'entropie
        # min_entropie : la valeur de l'entropie minimale
        # Xbest_valeurs : la liste des valeurs que peut prendre l'attribut i_best
        #
        # Il est donc nécessaire ici de parcourir tous les attributs et de calculer
        # la valeur de l'entropie de la classe pour chaque attribut.
        X_entropies = []
        
        for i in range(X.shape[1]): # pour chaque attribut Xj qui décrit les exemples de X
            # pour chacune des valeurs vjl de Xj construire l'ensemble des exemples de X qui possède la valeur vjl 
            s = pd.Series(X[:, i]) # ith column
            d = s.groupby(s).groups # dict[value = vjl] = indices ( a list Int64 index)
            
            vjls_ent = []
            # pour chacune des valeurs vjl de Xj
            for vjl, indices in d.items():
                vjl_labels = Y[indices] # ainsi que l'ensemble de leurs labels.
                
                vjl_entropie = entropie(vjl_labels) # HS(Y|vjl) 
                p_vjl = len(indices) / len(s)
                vjls_ent.append(vjl_entropie * p_vjl)
                
            # HS(Y|Xj) 
            Xj_entropie = sum(vjls_ent)
            
            X_entropies.append(Xj_entropie)
            
        
        min_entropie = min(X_entropies)
        i_best = X_entropies.index(min_entropie)
        Xbest_valeurs = np.unique(X[:, i_best])
        
        ############################# FIN ######## 
        
        if len(LNoms)>0:  # si on a des noms de features
            noeud = NoeudCategoriel(i_best,LNoms[i_best])    
        else:
            noeud = NoeudCategoriel(i_best)

        print(len(X))
        print(X)
        for v in Xbest_valeurs:
            noeud.ajoute_fils(v,construit_AD(X[X[:,i_best]==v], Y[X[:,i_best]==v],epsilon,LNoms))
    return noeud


# -*- coding: utf-8 -*-

"""
Package: iads
File: Classifiers.py
Année: LU3IN026 - semestre 2 - 2020-2021, Sorbonne Université
"""

# Classfieurs implémentés en LU3IN026
# Version de départ : Février 2021

# Import de packages externes
import math
import numpy as np
import pandas as pd
import random
from copy import copy, deepcopy
import copy
# ---------------------------
class Classifier:
    """ Classe pour représenter un classifieur
        Attention: cette classe est une classe abstraite, elle ne peut pas être
        instanciée.
    """
    #TODO: Classe à Compléter
    def __init__(self, input_dimension):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension de la description des exemples
            Hypothèse : input_dimension > 0
        """
        raise NotImplementedError("Please Implement this method")
        
    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        
        raise NotImplementedError("Please Implement this method")
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        raise NotImplementedError("Please Implement this method")
    
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        raise NotImplementedError("Please Implement this method")

    def accuracy(self, desc_set, label_set):
        """ Permet de calculer la qualité du système sur un dataset donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        # COMPLETER CETTE FONCTION ICI : 
        # ............
        # ............
        cpt = 0
        for i in range(desc_set.shape[0]):
            if self.predict(desc_set[i,:]) == label_set[i]:
                cpt = cpt + 1
        return cpt/desc_set.shape[0]
# ---------------------------
class ClassifierLineaireRandom(Classifier):
    """ Classe pour représenter un classifieur linéaire aléatoire
        Cette classe hérite de la classe Classifier
    """
    
    def __init__(self, input_dimension):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension de la description des exemples
            Hypothèse : input_dimension > 0
        """
        self.input_dimension = input_dimension
        #chaque composante de w est choisi entre -1 et 1
        self.w = 2*np.random.random(input_dimension) - 1
        self.w = self.w / np.linalg.norm(self.w)
        
    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        print("Pas d'apprentissage pour ce classifieur")
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        return np.dot(x,self.w)
    
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        if(self.score(x)<0):
            return -1
        return 1    
# ---------------------------
class ClassifierKNN(Classifier):
    """ Classe pour représenter un classifieur par K plus proches voisins.
        Cette classe hérite de la classe Classifier
    """
    #TODO: Classe à Compléter
    def __init__(self, input_dimension, k):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension d'entrée des exemples
                - k (int) : nombre de voisins à considérer
            Hypothèse : input_dimension > 0
        """
        self.input_dimension = input_dimension
        self.k = k
        
    def score(self,x):
        """ rend la proportion de +1 parmi les k ppv de x (valeur réelle)
            x: une description : un ndarray
        """
        size = self.train_desc_set.shape[0]
        #calcule le tableau de distances
        dist = np.zeros(size)
        for i in range(size):
            dist[i] = np.linalg.norm(self.train_desc_set[i] - x)
        #selection des k plus proches voisins
        indice = np.argsort(dist)[:self.k]
        #calcule du score
        pos = 0
        m = 0
        for i in indice:
            if self.train_label_set[i] > 0:
                pos+=1
            else:
                m+=1
        return (pos - m)/(pos + m)
    
    def predict(self, x):
        """ rend la prediction sur x (-1 ou +1)
            x: une description : un ndarray
        """
        return np.sign(self.score(x))

    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        

        self.train_desc_set = desc_set
        self.train_label_set = label_set
        
 # ---------------------------
class ClassifierPerceptron(Classifier):
    """ Perceptron de Rosenblatt
    """
    def __init__(self, input_dimension,learning_rate, history=False):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples
                - learning_rate : epsilon
            Hypothèse : input_dimension > 0
        """
        self.history = history 
        if(input_dimension!= None):
            self.w = np.random.rand(input_dimension)
        else:
            self.w = 0
        if(self.history):
            self.allw = []
        self.learning_rate = learning_rate

        #Si history est à True alors on initialise le tableau de sauvegarde
        if(history == True):
            self.history = True
            self.allw  = []
        
    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donnée
            réalise une itération sur l'ensemble des données prises aléatoirement
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        
        ordre = np.arange(desc_set.shape[0])
        np.random.shuffle(ordre)
        for i in ordre:
            elem = desc_set[i]
            z = self.predict(elem)
            if z * label_set[i] <= 0:
                self.w += self.w * elem * label_set[i]
                self.w /= np.linalg.norm(self.w)

                if(self.history):
                    self.allw.append(self.w)
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        return np.dot(x, self.w)
    
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        return np.sign(self.score(x))
    def getW(self):
	    """ rend le vecteur de poids actuel du perceptron
	    """
	    return self.w 

 # ---------------------------


class ClassifierPerceptronBiais(Classifier):

    """ Perceptron de Rosenblatt
    """
    def __init__(self, input_dimension,learning_rate, history=False,):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples
                - learning_rate : epsilon
            Hypothèse : input_dimension > 0
        """
        self.history = history 
        if(input_dimension!= None):
            self.w = np.random.rand(input_dimension)
        else:
            self.w = 0
        if(self.history):
            self.allw = []
        self.learning_rate = learning_rate

        #Si history est à True alors on initialise le tableau de sauvegarde
        if(history == True):
            self.history = True
            self.allw  = []
        
    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            réalise une itération sur l'ensemble des données prises aléatoirement
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """       
        #Tirage aleatoire de l'index
        r = list(range(desc_set.shape[0]))
        random.shuffle(r)
        for i in r:
            x_i = desc_set[i]
            #print(x_i)
            #Verification d'une potentielle correction soit signe différent soit f(x)y <1
            if(self.predict(x_i)!=label_set[i] and abs(self.score(x_i) * label_set[i])<1):
                self.w = self.w + self.learning_rate*(label_set[i]-self.predict(x_i))*x_i
                
                if(self.history):
                    self.allw.append(self.w)
        return	
        
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        return np.dot(x,self.w)
    
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        return np.sign((self.score(x)))
 
    def getW(self):
	    """ rend le vecteur de poids actuel du perceptron
	    """
	    return self.w 

# ---------------------------


class Kernel():
    """ Classe pour représenter des fonctions noyau
    """
    def __init__(self, dim_in, dim_out):
        """ Constructeur de Kernel
            Argument:
                - dim_in : dimension de l'espace de départ (entrée du noyau)
                - dim_out: dimension de l'espace de d'arrivée (sortie du noyau)
        """
        self.input_dim = dim_in
        self.output_dim = dim_out
        
    def get_input_dim(self):
        """ rend la dimension de l'espace de départ
        """
        return self.input_dim

    def get_output_dim(self):
        """ rend la dimension de l'espace d'arrivée
        """
        return self.output_dim
    
    def transform(self, V):
        """ ndarray -> ndarray
            fonction pour transformer V dans le nouvel espace de représentation
        """        
        raise NotImplementedError("Please Implement this method")


class KernelBias(Kernel):
    """ Classe pour un noyau simple 2D -> 3D
    """
    def transform(self, V):
        """ ndarray de dim 2 -> ndarray de dim 3            
            rajoute une 3e dimension au vecteur donné
        """
        V_proj = np.append(V,np.ones((len(V),1)),axis=1)
        return V_proj

class KernelPoly(Kernel):
    def transform(self,V):
        """ ndarray de dim 2 -> ndarray de dim 6
            ...
        """
        #Note optimisation: ajouter des cases à un array numpy est moins efficace qu'add à une liste python basique
        ##TODO
        V_tranform = np.append(np.ones((1,1)),V)
        V_tranform = np.append(V_tranform,V[0]*V[0])
        V_tranform = np.append(V_tranform,V[1]*V[1])
        V_tranform = np.append(V_tranform,V[0]*V[1])
        return V_tranform


class ClassifierPerceptronKernel(Classifier):
    """ Perceptron utilisant un kernel
    """
    def __init__(self, input_dimension, learning_rate, noyau):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples (espace originel)
                - learning_rate : 
                - noyau : Kernel à utiliser
            Hypothèse : input_dimension > 0
        """
        self.w = np.zeros(input_dimension)
        self.e = learning_rate
        self.k = noyau
        
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        z = self.k.transform(x)
        res = np.dot(z, self.w)
        return res
    
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        return np.sign(self.score(x))

    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        
        ordre = np.arange(desc_set.shape[0])
        np.random.shuffle(ordre)
        for i in ordre:
            elem = desc_set[i]
            elem = self.k.transform(elem)
            z = np.dot(elem, self.w)
            if z * label_set[i] <= 0:
                self.w += self.e * elem * label_set[i]
                # La normalisation de w a été choisie pour garantir
                #que chaque modification de w est petite (de l'ordre de self.e) 
                #par rapport à la valeur précédente de w.
                self.w /= np.linalg.norm(self.w)
    
                
# ------------------------ 





class ClassifierMultiOAA(Classifier):
    """ Perceptron de ADALINE
    """
    def __init__(self, classif):
        """ Constructeur de Classifier
            Argument:
                - classif: Classfier binary
        """
        self.classif = classif
        self.classif_array = []
        #Tableau des scores
        self.score_array = []
        
        
    def train(self, desc_set, label_set, n_copy = 10):
        """ Permet d'entrainer le modele sur l'ensemble donné
            réalise une itération sur l'ensemble des données prises aléatoirement
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """     
        #On copie NCI fois le nouveau array dans le tableau
        i=0
        while(i <= n_copy):
            self.classif_array.append(deepcopy(self.classif))
            i+=1  
        #Tirage aleatoire de l'index
        r = list(range(desc_set.shape[0]))
        random.shuffle(r)
        #Récupération des différentes classes possibles
        #list_classe = np.unique(label_set)
        
        #Parcours des classes
        for classif in self.classif_array:
            
                #On appelle la fonction train pour chaque classifier présent dans le tablea
            classif.train(desc_set,label_set)
                
                # for i in r:
                    
                #     x_i = desc_set[i]
                #     #Verification d'une potentielle correction soit signe différent soit f(x)y <1
                #     if(classif.predict(x_i)!=classe):
                #         #Calcul du gradient et affectation du nouveau w
                #         classif.w = classif.w - classif.learning_rate*(x_i.transpose()*x_i*classif.w - label_set[i])
                #         #On réindex si la classe est différente de "classe"
                #         label_set[index] = -1
                #         #if(self.history):
                #         #   self.allw.append(self.w)
                #     if(classif.predict(x_i)==classe):
                #         label_set[index] = 1
                        
                #     index += 1
                
        return
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        #On calcule le score et on ajoute tout ça dans un tableau
        score_array = []
        for classifier in self.classif_array:
            #On appelle la fonction score du classifier
            score_array.append(classifier.score(x))
        return score_array
    
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        #On retourne le argmax des scores calculés par les différents classifiers
        #Puis on renvoie le signe correspondant au résultat de la fonction score du classifieur
        #Ayant eu le meilleur résultat
        return np.argmax(self.score(x))
       
    def accuracy(self, desc_set, label_set):
        yhat = np.array([self.predict(x) for x in desc_set])
        return np.where(label_set == yhat, 1., 0.).mean()





class ClassifierADALINE(Classifier):
    """ Perceptron de ADALINE
    """
    #TODO: Classe à Compléter
    def __init__(self, input_dimension, learning_rate, history=False, niter_max=1000):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples
                - learning_rate : epsilon
                - history : stockage des poids w en cours d'apprentissage
                - niter_max : borne sur les iterations
            Hypothèse : input_dimension > 0
        """
        self.w = np.random.rand(input_dimension)
        self.learning_rate = learning_rate
        self.niter_max = niter_max
        
        if(history):
            self.allw = []
        self.learning_rate = learning_rate
        
        self.history = history
        #Si history est à True alors on initialise le tableau de sauvegarde
        if(self.history):
            self.allw  = []
        
        
    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            réalise une itération sur l'ensemble des données prises aléatoirement
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        
        #Tirage aleatoire de l'index
        r = list(range(desc_set.shape[0]))
        cpt = 0
        random.shuffle(r)
        for i in r:
            #Test du nombre max d'intération atteind
            if(cpt>= self.niter_max):
                return
            x_i = desc_set[i]
            #Verification d'une potentielle correction soit signe différent soit f(x)y <1
            
            #Calcul du gradient et affectation du nouveau w
            self.w = self.w - self.learning_rate*(x_i.transpose()*(x_i@self.w - label_set[i]))

            if(self.history):
                self.allw.append(self.w)
            
            cpt+=1
        return
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        return np.dot(x,self.w)
    
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        return np.sign((self.score(x)))
    
    
    def accuracy(self, desc_set, label_set):
        """ Permet de calculer la qualité du système sur un dataset donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        i = 0
        accuracy = np.zeros(len(desc_set))
        for x in desc_set:
            prediction = self.predict(x)
            if (prediction == label_set[i]):
                 accuracy[i] = 1
            else:
                accuracy[i] = 0
            i+=1
        return accuracy.mean()


class ClassifierADALINE2(Classifier):
    """ Perceptron de ADALINE
    """
    #TODO: Classe à Compléter
    def __init__(self, input_dimension):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples
            Hypothèse : input_dimension > 0
        """
        self.w = np.random.rand(input_dimension)
        
    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            réalise une résolution d'équation
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """    
        #On initialise nos np.array pour l'appelle de linalg.solve
        a = np.array(desc_set.transpose()@desc_set)
        #On retourne la solution de l'équation
        b = np.array(desc_set.transpose()@label_set)
        
        self.w= np.linalg.solve(a,b)
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        return np.dot(x,self.w)
    
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        return np.sign((self.score(x)))


####################################################
################## ARBRE DECISION ##################
####################################################

import graphviz as gv
def shannon(P):
    """ list[Number] -> float
        Hypothèse: la somme des nombres de P vaut 1
        P correspond à une distribution de probabilité
        rend la valeur de l'entropie de Shannon correspondante
    """
    #### A compléter pour répondre à la question posée
    k = len(P)
    if k <= 1:
        return 0.0
    
    return  -sum([i * math.log(i, k) if i > 0 else 0 for i in P])

def entropie(Y):
    """ Y : (array) : ensemble de labels de classe
        rend l'entropie de l'ensemble Y
    """
     # A COMPLETER
    labels = np.asarray(Y)
    
    valeurs, nb_fois = np.unique(labels ,return_counts=True)
    k = len(labels)
    
    return shannon([count_label / k for count_label in nb_fois])

def classe_majoritaire(Y):
    """ Y : (array) : array de labels
        rend la classe majoritaire ()
    """
    #### A compléter pour répondre à la question posée
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
        
        #############
        
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
            d = s.groupby(s).groups # dict[value = vjl] = indices ( liste Int64 index)
            
            vjls = []
            # pour chacune des valeurs vjl de Xj
            for vjl, indices in d.items():
                vjl_labels = Y[indices] # ainsi que l'ensemble de leurs labels.
                
                vjl_entropie = entropie(vjl_labels) # HS(Y|vjl) 
                p_vjl = len(indices) / len(s)
                vjls.append(vjl_entropie * p_vjl)
                
            # HS(Y|Xj) 
            Xj_entropie = sum(vjls)
            
            X_entropies.append(Xj_entropie)
            
        
        min_entropie = min(X_entropies)
        i_best = X_entropies.index(min_entropie)
        Xbest_valeurs = np.unique(X[:, i_best])
        
        ############
        
        if len(LNoms)>0:  # si on a des noms de features
            noeud = NoeudCategoriel(i_best,LNoms[i_best])    
        else:
            noeud = NoeudCategoriel(i_best)
        for v in Xbest_valeurs:
            noeud.ajoute_fils(v,construit_AD(X[X[:,i_best]==v], Y[X[:,i_best]==v],epsilon,LNoms))
    return noeud


class ClassifierArbreDecision(Classifier):
    """ Classe pour représenter un classifieur par arbre de décision
    """
    
    def __init__(self, input_dimension, epsilon, LNoms=[]):
        """ Constructeur
            Argument:
                - intput_dimension (int) : dimension de la description des exemples
                - epsilon (float) : paramètre de l'algorithme (cf. explications précédentes)
                - LNoms : Liste des noms de dimensions (si connues)
            Hypothèse : input_dimension > 0
        """
        self.dimension = input_dimension
        self.epsilon = epsilon
        self.LNoms = LNoms
        # l'arbre est manipulé par sa racine qui sera un Noeud
        self.racine = None
        
    def toString(self):
        """  -> str
            rend le nom du classifieur avec ses paramètres
        """
        return 'ClassifierArbreDecision ['+str(self.dimension) + '] eps='+str(self.epsilon)
        
    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        
        ##################
        ## COMPLETER ICI !
        self.racine = construit_AD(desc_set, label_set, self.epsilon, self.LNoms)
        ##################
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        # cette méthode ne fait rien dans notre implémentation :
        pass
    
    def predict(self, x):
        """ x (array): une description d'exemple
            rend la prediction sur x             
        """
        ##################
        ## COMPLETER ICI !
        root = deepcopy(self.racine)
        assert(len(self.LNoms) == len(x))
        
        try:
            while not root.est_feuille():
                root_index = list(self.LNoms).index(root.nom_attribut)
                root = deepcopy(root.Les_fils[x[root_index]])
        except KeyError:
            print("*** Warning: attribut " + str(root.nom_attribut) + " -> Valeur inconnue: " + str(x[root_index]))
            return 0 # inspired from below (la classification de certains exemples produit un warning)
            
        return root.classe
        ##################

    def affiche(self,GTree):
        """ affichage de l'arbre sous forme graphique
            Cette fonction modifie GTree par effet de bord
        """
        self.racine.to_graph(GTree)



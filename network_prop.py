"""
 -----------------------------------------------------------------------

Author: Brin Rosenthal (sbrosenthal@ucsd.edu)

 -----------------------------------------------------------------------
"""


#import matplotlib.pyplot as plt
#import seaborn
import networkx as nx
import pandas as pd
import random
import numpy as np
import itertools
#import json
import scipy
#import community
#from sklearn.cluster import AffinityPropagation
#from sklearn.cluster import AgglomerativeClustering


def normalized_adj_matrix(G,conserve_heat=True,weighted=False):
    
    '''
    This function returns normalized adjacency matrix.
    
    Inputs:
        - G: NetworkX graph from which to calculate normalized adjacency matrix
        - conserve_heat:
            - True: Heat will be conserved (sum of heat vector = 1).  Graph asymmetric
            - False:  Heat will not be conserved.  Graph symmetric.
    '''
    
    wvec=[]
    for e in G.edges(data=True):
        v1 = e[0]
        v2 = e[1]
        deg1 = G.degree(v1)
        deg2 = G.degree(v2)
        
        if weighted:
            weight = e[2]['weight']
        else:
            weight=1
        
        if conserve_heat:
            wvec.append((v1,v2,weight/float(deg2))) #np.sqrt(deg1*deg2)))
            wvec.append((v2,v1,weight/float(deg1)))
        else:
            wvec.append((v1,v2,weight/np.sqrt(deg1*deg2)))
    
    if conserve_heat:
        # if conserving heat, make G_weighted a di-graph (not symmetric)
        G_weighted= nx.DiGraph()
    else:
        # if not conserving heat, make G_weighted a simple graph (symmetric)
        G_weighted = nx.Graph()
        
    G_weighted.add_weighted_edges_from(wvec)
    
    Wprime = nx.to_numpy_matrix(G_weighted,nodelist=G.nodes())
    Wprime = np.array(Wprime)
    
    return Wprime

def network_propagation(G,Wprime,seed_genes,alpha=.5, num_its=20):
    
    '''
    This function implements network propagation, as detailed in:
    Vanunu, Oron, et al. 'Associating genes and protein complexes with disease via network propagation.'
    Inputs:
        - G: NetworkX graph on which to run simulation
        - Wprime:  Normalized adjacency matrix (from normalized_adj_matrix)
        - seed_genes:  Genes on which to initialize the simulation.
        - alpha:  Heat dissipation coefficient.  Default = 0.5
        - num_its:  Number of iterations (Default = 20.  Convergence usually happens within 10)
        
    Outputs:
        - Fnew: heat vector after propagation

    
    '''
    
    nodes = G.nodes()
    numnodes = len(nodes)
    edges=G.edges()
    numedges = len(edges)
    
    Fold = np.zeros(numnodes)
    Fold = pd.Series(Fold,index=G.nodes())
    Y = np.zeros(numnodes)
    Y = pd.Series(Y,index=G.nodes())
    for g in seed_genes:
        Y[g] = Y[g]+1/float(len(seed_genes)) # normalize total amount of heat added, allow for replacement
    Fold = Y.copy(deep=True)

    for t in range(num_its):
        Fnew = alpha*np.dot(Wprime,Fold) + np.multiply(1-alpha,Y)
        Fold=Fnew

    return Fnew
    


# function to return significant digits in exp form
def nsf(num, n=1):
    """n-Significant Figures"""
    numstr = ("{0:.%ie}" % (n-1)).format(num)
    return float(numstr)





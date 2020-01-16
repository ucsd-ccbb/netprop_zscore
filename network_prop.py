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
    

def calc_localization(Gint,genes_focal,write_file_name='localization_results',num_reps=5,num_genes=20,
                     conserve_heat=True, replace=True,subsample=True, savefile=True):
    
    seed_FOCAL = list(np.intersect1d(list(genes_focal),Gint.nodes()))
    
    if subsample:
        num_genes_S=num_genes
    else:
        num_genes_S=len(seed_FOCAL)

    Wprime = normalized_adj_matrix(Gint,conserve_heat=conserve_heat)
    
    kurt_FOCAL =[]
    kurt_Srand=[]
    var_FOCAL, var_Srand=[],[]
    sumTop_FOCAL= []
    sumTop_Srand= []
    for r in range(num_reps):
        print(r)
        
        subset_FOCAL = np.random.choice(seed_FOCAL,size=num_genes_S,replace=replace)

        Fnew_FOCAL = network_propagation(Gint,Wprime,subset_FOCAL,alpha=.5,num_its=20)
        Fnew_FOCAL.sort()
        kurt_FOCAL.append(scipy.stats.kurtosis(Fnew_FOCAL))
        var_FOCAL.append(np.var(Fnew_FOCAL))
        sumTop_FOCAL.append(np.sum(Fnew_FOCAL.head(1000)))

        G_temp = nx.configuration_model(Gint.degree().values())
        G_rand = nx.Graph()  # switch from multigraph to digraph
        G_rand.add_edges_from(G_temp.edges())
        # remove self-loops
        #G_rand.remove_edges_from(G_rand.selfloop_edges())
        G_rand = nx.relabel_nodes(G_rand,dict(zip(range(len(G_rand.nodes())),Gint.degree().keys())))
        Wprime_rand = normalized_adj_matrix(G_rand,conserve_heat=conserve_heat)

        Fnew_Srand = network_propagation(G_rand,Wprime_rand,subset_FOCAL,alpha=.5,num_its=20)
        Fnew_Srand.sort()
        kurt_Srand.append(scipy.stats.kurtosis(Fnew_Srand))
        var_Srand.append(np.var(Fnew_Srand))
        sumTop_Srand.append(np.sum(Fnew_Srand.head(1000)))

        print(var_FOCAL[-1])
        print(var_Srand[-1])
    
    results_dict = {'kurtosis':kurt_FOCAL,'kurt_rand':kurt_Srand,
                   'var':var_FOCAL,'var_rand':var_Srand,
                   'sumTop':sumTop_FOCAL, 'sumTop_rand':sumTop_Srand,
                   'num_reps':num_reps, 'conserve_heat':conserve_heat,
                   'replace':replace,'subsample':subsample,'num_genes':num_genes}
    
    if savefile:
        json.dump(results_dict,open(write_file_name,'w'))
        
    return results_dict



def calc_pos_labels(pos,dx=.03):
    
    '''
    Helper function to return label positions offset by dx
    
    - input node positions from nx.spring_layout()
    
    '''
    
    pos_labels = dict()
    for key in pos.keys():
        pos_labels[key] = np.array([pos[key][0]+dx,pos[key][1]+dx])
    
    return pos_labels


# function to return significant digits in exp form
def nsf(num, n=1):
    """n-Significant Figures"""
    numstr = ("{0:.%ie}" % (n-1)).format(num)
    return float(numstr)





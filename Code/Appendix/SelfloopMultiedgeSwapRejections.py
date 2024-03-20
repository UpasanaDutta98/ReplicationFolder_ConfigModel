import networkx as nx
import copy
import numpy as np
import math
from scipy import stats
import pandas as pd
import time
import numba as nb
import os
import pickle
import matplotlib.pyplot as plt
import time
from numba.typed import List

jit =  nb.jit

@jit(nopython=True,nogil=True)
def convert_edgelist_to_AdjMatrix(edgeList, A):
    for edge in edgeList:
        A[edge[0], edge[1]] += 1
        A[edge[1], edge[0]] += 1     

def read_edgelist(file):
    List_of_all_lines = file.readlines()
    edgelist = []
    
    for eachline in List_of_all_lines:
        eachline = eachline.strip()
        stripList = eachline.split(" ")
        edgelist.append((int(stripList[0]), int(stripList[1])))

    return edgelist

def create_graph_using_edgelist(edgelist, multi = 0):
    if multi == 1:
        G = nx.MultiGraph()
    else:
        G = nx.Graph()
    for eachedge in edgelist:
        G.add_edge(eachedge[0], eachedge[1])

    return G


#If numba can't load, the following do-nothing decorator will apply
def __no_jit__(*dargs,**dkwargs):
    def decorate(func):
        def call(*args,**kwargs):
            return func(*args,**kwargs)
        return call
    return decorate


@jit(nopython=True,nogil=True)
def MCMC_step_stub_withNodeList(A,edge_list,swaps,rejected,nodeList,allow_loops,allow_multi):
    '''
    
    Performs a stub-labeled double edge swap.
    
    | Args:
    |     A (nxn numpy array): The adjacency matrix. Will be changed inplace.
    |     edge_list (nx2 numpy array): List of edges in A. Node names should be 
            the integers 0 to n-1. Will be changed inplace. Edges must appear
            only once.
    |     swaps (length 4 numpy array): Changed inplace, will contain the four
            nodes swapped if a swap is accepted.
    |     allow_loops (bool): True only if loops allowed in the graph space.
    |     allow_multi (bool): True only if multiedges are allowed in the graph space.
    
    | Returns:
    |     bool: True if swap is accepted, False if current graph is resampled. 
    
    Notes
    -----
    This method currently requires a full adjacency matrix. Adjusting this 
    to work a sparse adjacency matrix simply requires removing the '@nb.jit'
    decorator. This method supports loopy graphs, but depending on the degree
    sequence, it may not be able to sample from all loopy graphs.
        
    '''
    # Choose two edges uniformly at random
    m= len(edge_list)
    p1 = np.random.randint(m)
    p2 = np.random.randint(m-1)
    if p1 == p2: # Prevents picking the same edge twice
        p2 = m-1
        
    u,v = edge_list[p1]        
    if np.random.rand()<0.5: #Pick either swap orientation 50% at random
        x,y = edge_list[p2]        
    else:
        y,x = edge_list[p2]

    # Note: tracking edge weights is the sole reason we require the adj matrix.
    # Numba doesn't allow sparse or dict objs. If you don't want to use numba
    # simply insert your favorite hash map (e.g. G[u][v] for nx multigraph G).
    w_ux = A[u,x]
    w_vy = A[v,y]

    nodeList[0] = u
    nodeList[1] = v
    nodeList[2] = x
    nodeList[3] = y
    # If multiedges are not allowed, resample if swap would replicate an edge
    if not allow_multi:
        if ( w_ux>=1 or w_vy>=1 ):
            # rejected[0] = 1 CHANGED HERE
            rejected[0] = 2
            return False
            
        if u == v and x == y:
            rejected[0] = 1
            return False
    
    #If loops are not allowed then only swaps on 4 distinct nodes are possible
    if not allow_loops:
        if u == x or u == y or v == x or v == y:
            # rejected[0] = 1 CHANGED HERE
            rejected[0] = 3
            return False
    
   
    swaps[0] = u # Numba currently is having trouble with slicing
    swaps[1] = v
    swaps[2] = x
    swaps[3] = y
    
    A[u,v] += -1
    A[v,u] += -1
    A[x,y] += -1
    A[y,x] += -1
    
    A[u,x] += 1
    A[x,u] += 1
    A[v,y] += 1
    A[y,v] += 1   

    edge_list[p1] = u,x
    edge_list[p2] = v,y
    
    return True
    
    
def check_rejection(G):   
    n = G.number_of_nodes()
    m = G.number_of_edges()
    rejected = np.zeros(1,dtype=np.int64)
    nodeList = np.zeros(4,dtype=np.int64)
    List_edges = []
    for edge in G.edges():                    
        List_edges.append(edge)                    
    edge_list = np.array(List_edges)

    A = np.zeros(shape=(n,n)) 
    convert_edgelist_to_AdjMatrix(edge_list, A)
    swaps = np.zeros(4,dtype=np.int64)
    
    # First we run the burn-in period of 1000m swaps.

    step_function = MCMC_step_stub_withNodeList
        
    for j in range(1000*m): 
        step_function(A, edge_list, swaps, rejected, nodeList, allow_loops=False, allow_multi=False)
    
    rejection_count_multiedge = 0
    rejection_count_loop = 0
    rejection_count_other = 0
    non_adjacent_count = 0
    
    # Then over a span of 1000m swaps, we note how many swaps are rejected due to multi-edge, due to self-loops, due to other.
    span = 1000*m
    for j in range(span):
        rejected[0] = 0   
        nodeList = np.zeros(4,dtype=np.int64)
        step_function(A, edge_list, swaps, rejected, nodeList, allow_loops=False, allow_multi=False)
        if rejected[0] == 2:
            rejection_count_multiedge += 1
        elif rejected[0] == 3:
            rejection_count_loop += 1
        elif rejected[0] == 1:
            rejection_count_other += 1
        number_unique_nodes = len(set(nodeList))
        if number_unique_nodes == 4:
            non_adjacent_count += 1
    del(A)
    final_G = nx.Graph()
    final_G.add_edges_from(edge_list)
    return rejection_count_multiedge/(span), rejection_count_loop/(2*span), rejection_count_other, non_adjacent_count/span, final_G # 2 here because when only 3 distinct nodes are chosen, only half of the time the swap is rejected due to introduction of self-loop. Other half is accepted.

def run():
    names = []
    edges = []
    nodes = []
    density = []
    loop_rejection_rate = []
    multiedge_rejection_rate = []
    other_rejection_rate = []
    non_adjacent_rate = []
    
    file1 = open('../../NetworkRepository/SimpleNetworkNames.txt', "r") # Read all simple networks.
    Allfiles = file1.readlines()
    
    for eachline in Allfiles:  
        textfileName = eachline.strip()
        file1 = open('../../NetworkRepository/SimpleNetworks/'+str(textfileName), "r")
        edgelist = read_edgelist(file1)
        G = create_graph_using_edgelist(edgelist, multi = 0)
        G3 = nx.convert_node_labels_to_integers(G)  
        m = G3.number_of_edges()
        n = G3.number_of_nodes()
        rho = (2*m)/(n*(n-1))
        edges.append(m)
        nodes.append(n)
        density.append(rho)
        names.append(textfileName[:-4])
        rejection_rates = check_rejection(G3) # Simple network
        # Note that for simple networks, is_vertex_labeled=False is same as is_vertex_labeled=True (identical graph space).
        multiedge_rejection_rate.append(rejection_rates[0])
        loop_rejection_rate.append(rejection_rates[1])
        other_rejection_rate.append(rejection_rates[2])
        non_adjacent_rate.append(rejection_rates[3])
        
    return names, edges, nodes, density, multiedge_rejection_rate, loop_rejection_rate, other_rejection_rate, non_adjacent_rate

# Note: This experiment is run only for simple networks. We are not leveraging parallelization here.
# For the same reason, this file does not take the graph space as a command line argument.
if __name__ ==  '__main__':
    results = run()                         
    pickle_out = open("../../Output/SwapsRejectionRateForSimpleNetworks/SwapRejectionRate_SimpleNetworkResults.pkl","wb")
    pickle.dump(results, pickle_out)
    pickle_out.close()
import networkx as nx
import dbl_edge_mcmc as mcmc
import copy
import time
import numpy as np
import pickle
import multiprocessing
import pandas as pd
import numba as nb
import os
import sys

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

def convert_weighted_to_Nonweighted(G, multi):
#     print("Nodes = ", G.number_of_nodes())
#     print("Edges = ", G.number_of_edges())
    if multi == 1:
        G_new = nx.MultiGraph()
    else:
        G_new = nx.Graph()
    for node in G.nodes():
        G_new.add_node(node)
    for edge in G.edges():
        G_new.add_edge(edge[0], edge[1])

    #print("After un-weighted ----")
#     print("Nodes = ", G_new.number_of_nodes())
#     print("Edges = ", G_new.number_of_edges())
    return G_new


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


def get_r_and_G_after1000mSwaps(G, allow_loops, allow_multi, is_vertex_labeled):
    G_degree = list(nx.degree(G))
    m = G.number_of_edges()
    n = G.number_of_nodes()

    rejected = np.zeros(1,dtype=np.int64)
    List_edges = []
    for edge in G.edges():                    
        List_edges.append(edge)                    
    edge_list = np.array(List_edges)

    A = np.zeros(shape=(n,n)) 
    convert_edgelist_to_AdjMatrix(edge_list, A)
    swaps = np.zeros(4,dtype=np.int64)
    
    if is_vertex_labeled == True:
        step_function = mcmc.MCMC_step_vertex
    else:
        step_function = mcmc.MCMC_step_stub
        
    for j in range(1000*m):
        step_function(A, edge_list, swaps, rejected, allow_loops, allow_multi)
        

    S1 = 2*m
    S2 = 0
    S3 = 0
    for i in range(n):
        S2 += (G_degree[i][1])**2
        S3 += (G_degree[i][1])**3
        
    denominator = S1*S3 - (S2**2)
    SL = 0
    for e in edge_list:
        SL += 2*G_degree[e[0]][1]*G_degree[e[1]][1]
    numerator = S1*SL - (S2**2)
    r = float(numerator)/denominator

    if allow_multi == False:
        nG = nx.Graph()
    else:
        nG = nx.MultiGraph()
        
    for i in range(n):
        nG.add_node(i)
    nG.add_edges_from(edge_list)
    
    return r, nG

def get_one_KsScatterPoint(G, allow_loops, allow_multi, is_vertex_labeled):
    r_converged_distribution = []
    G_converged_samples = []
    for i in range(200):
        returned_after1000mSwaps = get_r_and_G_after1000mSwaps(G, allow_loops, allow_multi, is_vertex_labeled)
        r_converged_distribution.append(returned_after1000mSwaps[0])
        G_converged_samples.append(returned_after1000mSwaps[1])

    return r_converged_distribution, G_converged_samples

def save_pickles(G, graphname, allow_loops, allow_multi, is_vertex_labeled, graphspace):
    returnedLists = get_one_KsScatterPoint(G, allow_loops, allow_multi, is_vertex_labeled)
    if not os.path.isdir("../../Output/1000mNetworks/" + graphspace + "/"):
        os.makedirs("../../Output/1000mNetworks/" + graphspace + "/")
    pickleFile = "../../Output/1000mNetworks/" + graphspace + "/" + str(graphname)+".pkl"
    pickle_out = open(pickleFile,"wb")
    pickle.dump(returnedLists[1], pickle_out)
    pickle_out.close()
    outputList = [graphname, G.number_of_edges(), returnedLists[0]]
    if not os.path.isdir("../../Output/1000mNetworkStatistics/" + graphspace + "/"):
        os.makedirs("../../Output/1000mNetworkStatistics/" + graphspace + "/")
    pickleFile = "../../Output/1000mNetworkStatistics/" + graphspace + "/" + str(graphname)+".pkl"
    pickle_out = open(pickleFile,"wb")
    pickle.dump(outputList, pickle_out)
    pickle_out.close()


def save_1000m_data(G, graphname, graphspace):
    if "Simple" in graphspace:
        allow_loops=False
        allow_multi=False
        if graphspace == "SimpleStub":
            is_vertex_labeled=False
        else:
            is_vertex_labeled=True
    elif "LoopyOnly" in graphspace:
        allow_loops=True
        allow_multi=False
        if graphspace == "LoopyOnlyStub":
            is_vertex_labeled=False
        else:
            is_vertex_labeled=True
    elif "MultiLoopy" in graphspace:
        allow_loops=True
        allow_multi=True
        if graphspace == "MultiLoopyStub":
            is_vertex_labeled=False
        else:
            is_vertex_labeled=True
    elif "MultiOnly" in graphspace:
        allow_loops=False
        allow_multi=True
        if graphspace == "MultiOnlyStub":
            is_vertex_labeled=False
        else:
            is_vertex_labeled=True    

    save_pickles(G, graphname, allow_loops, allow_multi, is_vertex_labeled, graphspace)
 
    
def store_graphs_and_netStatistics(list_of_graphs, list_of_names, graphspace):
    manager = multiprocessing.Manager()
    lock = multiprocessing.Lock() # Using multi-processing to process multiple networks parallely.
    jobs = []
    for i in range(len(list_of_graphs)):
        G = list_of_graphs[i]
        graphname = list_of_names[i]
        p = multiprocessing.Process(target=save_1000m_data, args=(G, graphname, graphspace))
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()

def run(graphspace):
    if graphspace == "SimpleStub" or graphspace == "SimpleVertex":
        file1 = open('../../NetworkRepository/SimpleNetworkNames.txt', "r")
    elif graphspace == "LoopyOnlyStub" or graphspace == "LoopyOnlyVertex":
        file1 = open('../../NetworkRepository/LoopyOnlyNetworkNames.txt', "r")
    elif graphspace == "MultiLoopyStub" or graphspace == "MultiLoopyVertex":
        file1 = open('../../NetworkRepository/MultiLoopyNetworkNames.txt', "r")
    elif graphspace == "MultiOnlyStub" or graphspace == "MultiOnlyVertex":
        file1 = open('../../NetworkRepository/MultiOnlyNetworkNames.txt', "r")
    
    Allfiles = file1.readlines()
    list_of_graphs = []
    list_of_names = []
    
    for eachline in Allfiles:
        if eachline == ".ipynb_checkpoints" or eachline == ".DS_Store":
            continue
        textfileName = eachline.strip()
        if graphspace == "SimpleStub" or graphspace == "SimpleVertex":
            file1 = open('../../NetworkRepository/SimpleNetworks/'+str(textfileName), "r")
        elif graphspace == "LoopyOnlyStub" or graphspace == "LoopyOnlyVertex":
            file1 = open('../../NetworkRepository/LoopyOnlyNetworks/'+str(textfileName), "r")
        elif graphspace == "MultiLoopyStub" or graphspace == "MultiLoopyVertex":
            file1 = open('../../NetworkRepository/MultiLoopyNetworks/'+str(textfileName), "r")
        elif graphspace == "MultiOnlyStub" or graphspace == "MultiOnlyVertex":
            file1 = open('../../NetworkRepository/MultiOnlyNetworks/'+str(textfileName), "r")

        edgelist = read_edgelist(file1)
        if graphspace in ["SimpleStub", "SimpleVertex", "LoopyOnlyStub", "LoopyOnlyVertex"]:
            G = create_graph_using_edgelist(edgelist, multi = 0)
        else:
            G = create_graph_using_edgelist(edgelist, multi = 1)
        G3 = nx.convert_node_labels_to_integers(G)  
        m = G3.number_of_edges()

        list_of_graphs.append(G3)
        list_of_names.append(str(textfileName[:-4]))

    store_graphs_and_netStatistics(list_of_graphs, list_of_names, graphspace)

if __name__ ==  '__main__':
    graphspace = sys.argv[1]
    run(graphspace = graphspace)
# For stub-labeled and vertex-labeled simple graphs, use graphspace = "SimpleStub" and "SimpleVertex", respectively.
# For stub-labeled and vertex-labeled loopy graphs, use graphspace = "LoopyOnlyStub" and "LoopyOnlyVertex", respectively.
# For stub-labeled and vertex-labeled loopy multigraphs, use graphspace = "MultiLoopyStub" and "MultiLoopyVertex", respectively.
# For stub-labeled and vertex-labeled multigraphs, use graphspace = "MultiOnlyStub" and "MultiOnlyVertex", respectively.
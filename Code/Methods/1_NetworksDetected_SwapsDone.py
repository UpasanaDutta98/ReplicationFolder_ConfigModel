# Please finish running 0_1_Run_EstimatedSamplingGap.py before running this file since this file depends on those generated outputs. 

import networkx as nx
import dbl_edge_mcmc as mcmc
import scipy.stats as stats
import statistics
import time
import pickle
import pandas as pd
import sys
from arch.unitroot import DFGLS
import numpy as np
import multiprocessing
import copy
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

def create_graph_using_edgelist(edgelist, multi = 0):
    if multi == 1:
        G = nx.MultiGraph()
    else:
        G = nx.Graph()
    for eachedge in edgelist:
        G.add_edge(eachedge[0], eachedge[1])

    return G

def convert_weighted_to_Nonweighted(G, multi):
    if multi == 1:
        G_new = nx.MultiGraph() 
    else:
        G_new = nx.Graph() 
    for node in G.nodes():
        G_new.add_node(node)
    for edge in G.edges():
        G_new.add_edge(edge[0], edge[1])
      
    return G_new

def maxDegree_criterion_satisfied(G):
    m = G.number_of_edges()
    degrees = list(nx.degree(G))
    degreeList = []
    for eachtuple in degrees:
        degreeList.append(eachtuple[1])

    sorted_degrees = sorted(degreeList, reverse = True)
    if sorted_degrees[0]*sorted_degrees[0] <= 2*m/3:
        return 1
    else:
        return 0

    
def density_criterion_satisfied(G, allow_loops):
    m = G.number_of_edges()
    n = G.number_of_nodes()

    if allow_loops == True:
        rho = m/(n*n/2)
    else:
        rho = m/(n*(n-1)/2)
    d = 2*rho - rho*rho

    if d < 0.25:
        return 1
    else:
        return 0

def get_assortativities_and_graph(step_function, A, edge_list, swaps, allow_loops, allow_multi, is_vertex_labeled, window, S1, S2, S3, denominator, G_degree, n, m, last_r):
    t = 0
    r_datapoints = []
    rejected = np.zeros(1,dtype=np.int64)
    new_r = last_r[0]
    while t < window:
        rejected[0] = 0        
        step_function(A,edge_list,swaps,rejected,allow_loops,allow_multi)
        delta_r = 0
        new_swap_list = swaps
        if rejected[0] != 1: # A swap was performed
            numerator = G_degree[new_swap_list[0]][1]*G_degree[new_swap_list[2]][1] + G_degree[new_swap_list[1]][1]*G_degree[new_swap_list[3]][1] - G_degree[new_swap_list[0]][1]*G_degree[new_swap_list[1]][1] - G_degree[new_swap_list[2]][1]*G_degree[new_swap_list[3]][1]
            delta_r = 2*numerator*2*m/denominator

        new_r = new_r + delta_r
                
        r_datapoints.append(new_r)
        t+=1
    last_r[0] = new_r   
    
    return r_datapoints

def run_convergence_detection(G, graphname, sampling_gap, allow_loops, allow_multi, is_vertex_labeled, graphspace):
    m = G.number_of_edges()
    n = G.number_of_nodes()
    G_degree = list(nx.degree(G))
    Results = [graphname, m]
    
    S1 = 2*m
    S2 = 0
    S3 = 0
    for i in range(n):
        S2 += (G_degree[i][1])**2
        S3 += (G_degree[i][1])**3
        
    denominator = S1*S3 - (S2**2)
    SL = 0
    for e in G.edges():
        SL += 2*G_degree[e[0]][1]*G_degree[e[1]][1]
    numerator = S1*SL - (S2**2)
    r_initial = numerator/denominator
    last_r = [r_initial]
        
        
    if allow_multi == False:
        if density_criterion_satisfied(G, allow_loops) == 1:
            window = 2*m
        else:
            window = sampling_gap
    else:
        if is_vertex_labeled == False: 
            window = 2*m
        else:
            if maxDegree_criterion_satisfied(G) == 1:
                window = 2.3*m
            else:
                window = sampling_gap               
                
    swaps_done = []
    deg_assort_detected_samples = []
    detected_networks = []

    if is_vertex_labeled == True:
        step_function = mcmc.MCMC_step_vertex
    else:
        step_function = mcmc.MCMC_step_stub
        
    iterations = 0
    while iterations < 200:
        List_edges = []
        for edge in G.edges():                    
            List_edges.append(edge)                    
        edge_list = np.array(List_edges)

        A = np.zeros(shape=(n,n)) 
        convert_edgelist_to_AdjMatrix(edge_list, A)
        swaps = np.zeros(4,dtype=np.int64)
        last_r = [r_initial]
        
        found = 0
        countchecks = 0
        while found < 2:
            if found == 0:
                test_r = get_assortativities_and_graph(step_function, A, edge_list, swaps, allow_loops, allow_multi, is_vertex_labeled, window, S1, S2, S3, denominator, G_degree, n, m, last_r)
            elif found == 1:
                incrementwindow = 2*m
                test_r = get_assortativities_and_graph(step_function, A, edge_list, swaps, allow_loops, allow_multi, is_vertex_labeled, incrementwindow, S1, S2, S3, denominator, G_degree, n, m, last_r)
                break
            try:
                result = DFGLS(test_r, trend = "c", lags=0)
                countchecks += 1
                if result.pvalue < 0.05: # Null: Non-stationarity. So reject non-stationarity.
                    found = 1
                else:
                    found = 0
            except:
                found = 999
                break
                
        if found == 999:
            iterations = 200

        del(result)  
        
        swaps_done.append((window*countchecks) + 2*m)
        r = test_r[-1]
        deg_assort_detected_samples.append(r)
        if allow_multi == False:
            nG = nx.Graph()
        else:
            nG = nx.MultiGraph() 
        n = G.number_of_nodes()
        for i in range(n):
            nG.add_node(i)
        for eachedge in edge_list:
            nG.add_edge(eachedge[0], eachedge[1])
        
        detected_networks.append(nG)
        del(nG)
        iterations+= 1 
    
    Results_Net = [graphname, m, detected_networks, swaps_done] # NetworksDetectedAtConvergence
    if not os.path.isdir("../../Output/NetworksDetectedAtConvergence/" + graphspace + "/"):
        os.makedirs("../../Output/NetworksDetectedAtConvergence/" + graphspace + "/")
    pickleFile = "../../Output/NetworksDetectedAtConvergence/" + graphspace + "/" + str(graphname)+".pkl"
    pickle_out = open(pickleFile,"wb")
    pickle.dump(Results_Net, pickle_out)
    pickle_out.close()
    
def save_averageSwaps_data(G, graphname, sampling_gap, graphspace):
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

    run_convergence_detection(G, graphname, sampling_gap, allow_loops, allow_multi, is_vertex_labeled, graphspace)

# Estimated Sampling gaps of networks are needed to use as the window-size while detecting convergence using DFGLS test.
def store_averageSwapsToConvergence(list_of_graphs, list_of_names, graphspace):
    manager = multiprocessing.Manager() # Using multi-processing to process multiple networks parallely.
    jobs = []
    if not os.path.exists("../../Output/EstimatedSamplingGaps/" + graphspace + "_SamplingGaps.csv"):
        raise ValueError("Sampling gap csv-file (pre-required) for graphspace " + graphspace + " does not exist. Generate that first.")
    Gap_csv = pd.read_csv("../../Output/EstimatedSamplingGaps/" + graphspace + "_SamplingGaps.csv")
    gap_dict = {} # Generate a dictionary with key = network name and value = sampling gap
    Network = list(Gap_csv['Network'])
    Gap = list(Gap_csv['gap'])
    for i in range(len(Network)):
        gap_dict[Network[i]] = float(Gap[i])
    
    for i in range(len(list_of_graphs)):
        G = list_of_graphs[i]
        graphname = list_of_names[i]
        if graphname not in gap_dict:
            raise ValueError("Sampling gap for network " + graphname + "not found in the csv file. Exiting.")
        sampling_gap = gap_dict[graphname]
        p = multiprocessing.Process(target=save_averageSwaps_data, args=(G, graphname, sampling_gap, graphspace))
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

    store_averageSwapsToConvergence(list_of_graphs, list_of_names, graphspace)


if __name__ ==  '__main__':
    graphspace = sys.argv[1]
    run(graphspace = graphspace)
# For stub-labeled and vertex-labeled simple graphs, use graphspace = "SimpleStub" and "SimpleVertex", respectively.
# For stub-labeled and vertex-labeled loopy graphs, use graphspace = "LoopyOnlyStub" and "LoopyOnlyVertex", respectively.
# For stub-labeled and vertex-labeled loopy multigraphs, use graphspace = "MultiLoopyStub" and "MultiLoopyVertex", respectively.
# For stub-labeled and vertex-labeled multigraphs, use graphspace = "MultiOnlyStub" and "MultiOnlyVertex", respectively.
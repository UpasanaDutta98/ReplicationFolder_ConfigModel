# Please finish running 0_1_Run_EstimatedSamplingGap.py, 0_2_Store_1000mGraphs.py, and 1_1_AverageSwapsToConvergence.py before running this file since this file depends on their generated outputs.

import igraph as ig
import networkx as nx
import pickle
import pandas as pd
import numpy as np
import time
import scipy.stats as stats
from collections import defaultdict
import multiprocessing
import sys
import os

def get_igraph_network(G):
    n = G.number_of_nodes()
    iG = ig.Graph(directed=False)
    iG.add_vertices(n)
    edgeList = list(G.edges())
    iG.add_edges(edgeList)
    return iG

def get_degree_assortativity(G):
    G_degree = list(nx.degree(G))
    m = G.number_of_edges()
    n = G.number_of_nodes()
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
    r = float(numerator)/denominator
    return r

def read_edgelist(file):
    List_of_all_lines = file.readlines()
    edgelist = []
    
    for eachline in List_of_all_lines:
        eachline = eachline.strip()
        stripList = eachline.split(" ")
        edgelist.append((int(stripList[0]), int(stripList[1])))

    return edgelist

def create_multigraph_using_edgelist(edgelist):
    G = nx.MultiGraph()
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

def get_weighted_network(G): # G is a MultiGraph, this is required for calculating edge connectivity when allow_multi = True
    n = G.number_of_nodes()
    iG = ig.Graph(directed=False)
    iG.add_vertices(n)
    edgeList = list(G.edges())
    iG.add_edges(edgeList)
    weights_of_edges = defaultdict(int)
    for eachedge in G.edges():
        n1 = eachedge[0]
        n2 = eachedge[1]
        if n1 > n2:
            t = n1
            n1 = n2
            n2 = t
        weights_of_edges[(n1, n2)] += 1
    
    weights_for_SimpleNetwork = []
    for e in iG.es:
        weight = weights_of_edges[e.tuple]
        weights_for_SimpleNetwork.append(weight)
    iG.es['weight'] = weights_for_SimpleNetwork
    return iG

def get_weighted_clustering_coefficient(G):
    n = G.number_of_nodes()
    edgeList = list(G.edges())
    weighted_edgeList = defaultdict(int)
    for edge in edgeList:
        n1 = edge[0]
        n2 = edge[1]
        if n1 > n2:
            t = n1
            n1 = n2
            n2 = t
        weighted_edgeList[(n1, n2)] += 1
            
    # Maybe keep the self-loops and check once?
    Gnew = nx.Graph()
    for edge in weighted_edgeList:
        Gnew.add_edge(edge[0], edge[1], weight = weighted_edgeList[(edge[0], edge[1])])    
    weighted_cc = np.mean(list(dict(nx.clustering(Gnew, weight = "weight")).values()))
    return weighted_cc

def get_weighted_diameter(iG):
    dia = iG.diameter(directed=False)
    return dia

def get_weighted_average_path_length(iG):
    mgp = iG.average_path_length(directed=False)
    return mgp

def get_unweighted_r_cc_dia_mgp(G): # This is for Simple and Loopy graphs only
    r = get_degree_assortativity(G)
    n = G.number_of_nodes()
    iG = ig.Graph(directed=False)
    iG.add_vertices(n)
    edgeList = G.edges()
    iG.add_edges(edgeList)
    cc = iG.transitivity_undirected()
    dia = iG.diameter(directed=False)
    mgp = iG.average_path_length(directed=False)
    return r, cc, dia, mgp

def get_weighted_numTriangles(G):
    n = G.number_of_nodes()
    edgeList = list(G.edges())
    weighted_edgeList = defaultdict(int)
    for edge in edgeList:
        n1 = edge[0]
        n2 = edge[1]
        weighted_edgeList[(n1, n2)] += 1
        if n1 != n2: # If it is a self-loop, both-way counting will lead to double-counting of the loop weight, which can affect the max_weight calculation.
            weighted_edgeList[(n2, n1)] += 1
    max_weight = max(dict(weighted_edgeList).values())
    
    AdjList = [set() for k in range(n)]
    NeighbourList = [set() for k in range(n)]
    for edge in weighted_edgeList:
        n1 = edge[0]
        n2 = edge[1]
        if n1 != n2: # Because self-loops should not be counted while counting triangles, so self will not be self's neighbour.
            w = weighted_edgeList[(n1, n2)]
            AdjList[n1].add((n2, w))
            AdjList[n2].add((n1, w))
            NeighbourList[n1].add(n2)
            NeighbourList[n2].add(n1)
    
    numT_sum_all_nodes = 0

    for n in range(G.number_of_nodes()):
        numT_of_this_node = 0
        neighbourSet_with_weights = list(AdjList[n])
        if len(neighbourSet_with_weights) > 1:
            for i in range(len(neighbourSet_with_weights)-1):
                neighbour1, weight1 = neighbourSet_with_weights[i][0], neighbourSet_with_weights[i][1]
                for j in range(i+1, len(neighbourSet_with_weights)):
                    neighbour2, weight2 = neighbourSet_with_weights[j][0], neighbourSet_with_weights[j][1]
                    n1 = neighbour1
                    n2 = neighbour2
                    if n2 in NeighbourList[n1]:
                        weight3 = weighted_edgeList[(n1, n2)]
                        numT_of_this_node += (((weight1/max_weight) * (weight2/max_weight) * (weight3/max_weight))**(1/3))
        numT_sum_all_nodes += numT_of_this_node

    return numT_sum_all_nodes/3

def get_weighted_numSquares(G):
    n = G.number_of_nodes()
    edgeList = list(G.edges())
    weighted_edgeList = defaultdict(int)
    for edge in edgeList:
        n1 = edge[0]
        n2 = edge[1]
        weighted_edgeList[(n1, n2)] += 1
        if n1 != n2: # If it is a self-loop, both-way counting will lead to double-counting of the loop weight, which can affect the max_weight calculation.
            weighted_edgeList[(n2, n1)] += 1
    max_weight = max(dict(weighted_edgeList).values())
    
    AdjList = [set() for k in range(n)]
    NeighbourList = [set() for k in range(n)]
    for edge in weighted_edgeList:
        n1 = edge[0]
        n2 = edge[1]
        if n1 != n2: # Because self-loops should not be counted while counting squares, so self will not be self's neighbour.
            w = weighted_edgeList[(n1, n2)]
            AdjList[n1].add((n2, w))
            AdjList[n2].add((n1, w))
            NeighbourList[n1].add(n2)
            NeighbourList[n2].add(n1)
        
    numS_sum_all_nodes = 0

    for n in range(G.number_of_nodes()):
        numS_of_this_node = 0
        neighbourSet_with_weights = list(AdjList[n])
        if len(neighbourSet_with_weights) > 1:
            for i in range(len(neighbourSet_with_weights)-1):
                neighbour1, weight1 = neighbourSet_with_weights[i][0], neighbourSet_with_weights[i][1]
                for j in range(i+1, len(neighbourSet_with_weights)):
                    neighbour2, weight2 = neighbourSet_with_weights[j][0], neighbourSet_with_weights[j][1]
                    common_neighbour_set = NeighbourList[neighbour1].intersection(NeighbourList[neighbour2])
                    common_neighbour_set.remove(n)
                    for commonNeighbour in common_neighbour_set:
                        weight3 = weighted_edgeList[(neighbour1, commonNeighbour)]
                        weight4 = weighted_edgeList[(neighbour2, commonNeighbour)]
                        numS_of_this_node += (((weight1/max_weight) * (weight2/max_weight) * (weight3/max_weight) * (weight4/max_weight))**(1/4))
        numS_sum_all_nodes += numS_of_this_node     
        
    return numS_sum_all_nodes/4

def get_numTriangles(G):
    # Number of all triangles.
    n = G.number_of_nodes()
    edgeList = list(G.edges())
    NeighbourList = [set() for k in range(n)]
    for edge in edgeList:
        n1 = edge[0]
        n2 = edge[1]
        if n1 != n2:
            NeighbourList[n1].add(n2)
            NeighbourList[n2].add(n1)
        
    number_of_all_triangles = 0

    for n in range(G.number_of_nodes()):
        neighbourSet = list(NeighbourList[n])
        if len(neighbourSet) > 1:
            for i in range(len(neighbourSet)-1):
                neighbour1 = neighbourSet[i]
                for j in range(i+1, len(neighbourSet)):
                    neighbour2 = neighbourSet[j]
                    if neighbour2 in NeighbourList[neighbour1]:
                        number_of_all_triangles += 1
                        
    return number_of_all_triangles//3

def get_numSquares(G):
    # Number of all squares
    n = G.number_of_nodes()
    edgeList = list(G.edges())
    NeighbourList = [set() for k in range(n)]
    for edge in edgeList:
        n1 = edge[0]
        n2 = edge[1]
        if n1 != n2:
            NeighbourList[n1].add(n2)
            NeighbourList[n2].add(n1)
        
    number_of_all_squares = 0

    for n in range(G.number_of_nodes()):
        neighbourSet = list(NeighbourList[n])
        if len(neighbourSet) > 1:
            for i in range(len(neighbourSet)-1):
                neighbour1 = neighbourSet[i]
                for j in range(i+1, len(neighbourSet)):
                    neighbour2 = neighbourSet[j]
                    common_neighbour_set = NeighbourList[neighbour1].intersection(NeighbourList[neighbour2])
                    common_neighbour_set.remove(n)
                    number_of_all_squares += len(common_neighbour_set)
                        
    return number_of_all_squares//4

def get_unweighted_edge_connectivity(G):
    iG = get_igraph_network(G)
    ec = iG.edge_connectivity()
    return ec

def get_weighted_edge_connectivity(G): 
    iG = get_weighted_network(G)  
    ec = iG.edge_connectivity()
    return ec

def get_radius(G): # Both simple and multi
    iG = get_igraph_network(G)
    rad = iG.radius()
    return rad

def test_convergence_all_statistics(graphname, allow_multi, graphspace):
    time1 = time.time()
    
    detected_r = []
    converged_r = []
    detected_cc = []
    converged_cc = []
    detected_dia = []
    converged_dia = []
    detected_mgp = []
    converged_mgp = []
    detected_numtriangles = []
    converged_numtriangles = []
    detected_numsquares = []
    converged_numsquares = []
    detected_edgeconnect = []
    converged_edgeconnect = []
    detected_rad = []
    converged_rad = []    

    if not os.path.exists("../../Output/NetworksDetectedAtConvergence/" + graphspace + "/" + str(graphname)+".pkl"):
        raise ValueError("Pre-required files for network " + graphname + " does not exist. Generate that first.")    
    pickleFile = "../../Output/NetworksDetectedAtConvergence/" + graphspace + "/" + str(graphname)+".pkl"
    pickle_in = open(pickleFile,"rb")
    savedFile = pickle.load(pickle_in)
    DetectedNetworks = savedFile[2]
    swapsdone = savedFile[3]
        
    # Iterate over each of the 200 networks at the point-of-convergence
    for eachNetwork in DetectedNetworks:
        G = nx.convert_node_labels_to_integers(eachNetwork) # Preserves the type of graph, MultiGraph/Graph
        if allow_multi == False: # For simple or loopy graphs
            r, cc, dia, mgp = get_unweighted_r_cc_dia_mgp(G)
            numT = get_numTriangles(G)
            numS = get_numSquares(G)
            EC = get_unweighted_edge_connectivity(G)
            rad = get_radius(G)            
        else: # For multigraphs or loopy-multigraphs
            r = get_degree_assortativity(G)
            cc = get_weighted_clustering_coefficient(G)
            iG = get_igraph_network(G)
            dia = get_weighted_diameter(iG)
            mgp = get_weighted_average_path_length(iG)
            numT = get_weighted_numTriangles(G)
            numS = get_weighted_numSquares(G)
            EC = get_weighted_edge_connectivity(G)
            rad = get_radius(G)
            
        detected_r.append(r)
        detected_cc.append(cc)
        detected_dia.append(dia)
        detected_mgp.append(mgp)
        detected_numtriangles.append(numT)
        detected_numsquares.append(numS)
        detected_edgeconnect.append(EC)
        detected_rad.append(rad)

    del(DetectedNetworks)
    del(savedFile)
    del(pickle_in)
    
    # Now read the stored networks obtained after applying 1000*m double-edge-swaps to the empirical networks.
    if not os.path.exists("../../Output/1000mNetworks/" + graphspace + "/" + str(graphname)+".pkl"):
        raise ValueError("Pre-required files for network " + graphname + " does not exist. Generate that first.") 
    pickleFile = "../../Output/1000mNetworks/" + graphspace + "/" + str(graphname)+".pkl"
    pickle_in = open(pickleFile,"rb")
    DetectedNetworks1000m = pickle.load(pickle_in)
    
    # Iterate over each of the 200 networks obtained after applying 1000*m swaps 
    for eachNetwork in DetectedNetworks1000m:
        G = nx.convert_node_labels_to_integers(eachNetwork)  
        if allow_multi == False:
            r, cc, dia, mgp = get_unweighted_r_cc_dia_mgp(G)
            numT = get_numTriangles(G)
            numS = get_numSquares(G)
            EC = get_unweighted_edge_connectivity(G)
            rad = get_radius(G)            
        else:
            r = get_degree_assortativity(G)
            cc = get_weighted_clustering_coefficient(G)
            iG = get_igraph_network(G)
            dia = get_weighted_diameter(iG)
            mgp = get_weighted_average_path_length(iG)
            numT = get_weighted_numTriangles(G)
            numS = get_weighted_numSquares(G)
            EC = get_weighted_edge_connectivity(G)
            rad = get_radius(G)
            
        converged_r.append(r)
        converged_cc.append(cc)
        converged_dia.append(dia)
        converged_mgp.append(mgp)
        converged_numtriangles.append(numT)
        converged_numsquares.append(numS)
        converged_edgeconnect.append(EC)
        converged_rad.append(rad)

    del(DetectedNetworks1000m)
    del(pickle_in)
        
    m = G.number_of_edges()
    KSstatistic1, p_value1 = stats.ks_2samp(detected_r,converged_r)
    r_results = ["r", detected_r, converged_r, KSstatistic1, p_value1]
    
    KSstatistic2, p_value2 = stats.ks_2samp(detected_cc,converged_cc)
    cc_results = ["CC", detected_cc, converged_cc, KSstatistic2, p_value2]
    
    KSstatistic3, p_value3 = stats.ks_2samp(detected_dia,converged_dia)
    dia_results = ["Dia", detected_dia, converged_dia, KSstatistic3, p_value3]
    
    KSstatistic4, p_value4 = stats.ks_2samp(detected_mgp,converged_mgp)
    mgp_results = ["Mgp", detected_mgp, converged_mgp, KSstatistic4, p_value4]
    
    KSstatistic5, p_value5 = stats.ks_2samp(detected_numtriangles,converged_numtriangles)
    numT_results = ["numT", detected_numtriangles, converged_numtriangles, KSstatistic5, p_value5]  
    
    KSstatistic6, p_value6 = stats.ks_2samp(detected_numsquares,converged_numsquares)
    numS_results = ["numS", detected_numsquares, converged_numsquares, KSstatistic6, p_value6]
    
    KSstatistic7, p_value7 = stats.ks_2samp(detected_edgeconnect,converged_edgeconnect)
    edgeconnect_results = ["EC", detected_edgeconnect, converged_edgeconnect, KSstatistic7, p_value7]
    
    KSstatistic8, p_value8 = stats.ks_2samp(detected_rad,converged_rad)
    rad_results = ["rad", detected_rad, converged_rad, KSstatistic8, p_value8]
    time2 = time.time()
    total_time = (time2-time1)/3600

    # Note: For network statistics that have a distribution with less than 10 unique values, we report chi-square test results in our manuscript (Figure 13 in appendix). The chi-sq results are computed when creating the plot. Here, we are saving the KS-test results just for consistency across the results stored for all network statistics.

    Results = [graphname, m, r_results, cc_results, dia_results, mgp_results, numT_results, numS_results, edgeconnect_results, rad_results, swapsdone, total_time]
    if not os.path.isdir("../../Output/All8NetworkStatisticsAtConvergence/" + graphspace + "/"):
        os.makedirs("../../Output/All8NetworkStatisticsAtConvergence/" + graphspace + "/")
    pickleFile = "../../Output/All8NetworkStatisticsAtConvergence/" + graphspace + "/" + str(graphname)+".pkl"
    pickle_out = open(pickleFile,"wb") 
    pickle.dump(Results, pickle_out)
    pickle_out.close()
    
def store_averageSwapsToConvergence(list_of_names, graphspace):
    manager = multiprocessing.Manager()
    jobs = []
    for i in range(len(list_of_names)):
        graphname = list_of_names[i]
        if graphspace in ["SimpleStub", "SimpleVertex", "LoopyOnlyStub", "LoopyOnlyVertex"]:
            allow_multi = False
        else:
            allow_multi = True
        p = multiprocessing.Process(target=test_convergence_all_statistics, args=(graphname, allow_multi, graphspace))
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
    list_of_names = []
    
    for eachline in Allfiles:
        if eachline == ".ipynb_checkpoints" or eachline == ".DS_Store":
            continue
        textfileName = eachline.strip()
        list_of_names.append(str(textfileName[:-4]))

    store_averageSwapsToConvergence(list_of_names, graphspace)

if __name__ ==  '__main__':
    graphspace = sys.argv[1]
    run(graphspace = graphspace)
# For stub-labeled and vertex-labeled simple graphs, use graphspace = "SimpleStub" and "SimpleVertex", respectively.
# For stub-labeled and vertex-labeled loopy graphs, use graphspace = "LoopyOnlyStub" and "LoopyOnlyVertex", respectively.
# For stub-labeled and vertex-labeled loopy multigraphs, use graphspace = "MultiLoopyStub" and "MultiLoopyVertex", respectively.
# For stub-labeled and vertex-labeled multigraphs, use graphspace = "MultiOnlyStub" and "MultiOnlyVertex", respectively.
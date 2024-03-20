# Please finish running 0_1_Run_EstimatedSamplingGap.py and 0_2_Store_1000mGraphs.py before running this file since this file depends on their generated outputs.

import networkx as nx
import dbl_edge_mcmc as mcmc
import scipy.stats as stats
import time
import pickle
import pandas as pd
import sys
import numpy as np
import multiprocessing
import copy
import numba as nb
import pymc.diagnostics
import os
import warnings

jit =  nb.jit
warnings.filterwarnings("ignore")

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


def run_RafteryLewis_tests(G, graphname, sampling_gap, graphspace, allow_loops, allow_multi, is_vertex_labeled):
    m = G.number_of_edges()
    n = G.number_of_nodes()
    G_degree = list(nx.degree(G))
    Results = [graphname, m]
    q = 0
    quantiles = []
    while q <= 1:
        quantiles.append(q)
        q = q + 0.1
        q = round(q, 1)
    
    time1 = time.time()
    
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
                window = 2*m
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
        while found < 1:
            test_r = get_assortativities_and_graph(step_function, A, edge_list, swaps, allow_loops, allow_multi, is_vertex_labeled, window, S1, S2, S3, denominator, G_degree, n, m, last_r)
            countchecks += 1
            max_burn_in = 0
            best_quantile = 0
            for quantile in quantiles:
                returned = pymc.diagnostics.raftery_lewis(test_r, q=quantile, r=0.005, s=.95, epsilon=.001, verbose=0)
                if returned[2] > max_burn_in:
                    max_burn_in = returned[2]
                    best_quantile = quantile

            converging_point_RafteryLewisTest = max_burn_in
            found = 1
            if converging_point_RafteryLewisTest < len(test_r):
                r = test_r[converging_point_RafteryLewisTest]
            else:
                rejected = np.zeros(1,dtype=np.int64)
                for i in range(max_burn_in - len(test_r)): # thats the excess we have yet to go
                    step_function(A,edge_list,swaps,rejected,allow_loops,allow_multi)
                if is_vertex_labeled == True:
                    G2 = nx.MultiGraph()
                else:
                    G2 = nx.Graph()
                for b in range(n):
                    G2.add_node(b)
                G2.add_edges_from(edge_list)
                r = get_degree_assortativity(G2)


        if found == 999:
            iterations = 200

        del(edge_list)
        del(A)

        swaps_done.append((window)*(countchecks))
        deg_assort_detected_samples.append(r)

        iterations+= 1


    if found == 999:
        windowSpecific_results = [window, np.inf, np.inf, np.inf, np.inf, np.inf] # (only for error-handling)
    else:
        mean_swaps = int(np.mean(swaps_done))
        # Now read the stored networks obtained after applying 1000*m double-edge-swaps to the empirical networks.
        if not os.path.exists("../../Output/1000mNetworks/" + graphspace + "/" + str(graphname)+".pkl"):
            raise ValueError("Pre-required files for network " + graphname + " does not exist. Generate that first.") 
        pickleFile = "../../Output/1000mNetworks/" + graphspace + "/" + str(graphname) + ".pkl"
        pickle_in = open(pickleFile,"rb")
        DetectedNetworks1000m = pickle.load(pickle_in)

        deg_assortativities_1000m = []
        for eachnet in DetectedNetworks1000m:
            deg_assortativities_1000m.append(get_degree_assortativity(eachnet))
        del(DetectedNetworks1000m)
        # Now perform KS-test between degree assortativity at point-of-convergence and degree assortativity at 1000m-swaps.
        KSstatistic, p_value = stats.ks_2samp(deg_assortativities_1000m,deg_assort_detected_samples)
        windowSpecific_results = [window, KSstatistic, p_value, deg_assortativities_1000m, deg_assort_detected_samples, mean_swaps, swaps_done]
    Results.append(windowSpecific_results)

    if not os.path.isdir("../../Output/OtherDiagnosticTests/RafteryLewis/" + graphspace + "/"):
        os.makedirs("../../Output/OtherDiagnosticTests/RafteryLewis/" + graphspace + "/")
    pickleFile = "../../Output/OtherDiagnosticTests/RafteryLewis/" + graphspace + "/" + str(graphname)+".pkl"
    pickle_out = open(pickleFile,"wb") 
    pickle.dump(Results, pickle_out)
    pickle_out.close()
    


def save_RafteryLewis_data(G, graphname, sampling_gap, graphspace):
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

    run_RafteryLewis_tests(G, graphname, sampling_gap, graphspace, allow_loops, allow_multi, is_vertex_labeled)

def store_RafteryLewisTestResults(list_of_graphs, list_of_names, graphspace):
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
            raise ValueError("Sampling gap for network " + graphname + "not found in csv file. Exiting.")
        sampling_gap = gap_dict[graphname]
        p = multiprocessing.Process(target=save_RafteryLewis_data, args=(G, graphname, sampling_gap, graphspace))
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

    store_RafteryLewisTestResults(list_of_graphs, list_of_names, graphspace)


if __name__ ==  '__main__':
    graphspace = sys.argv[1]
    run(graphspace = graphspace)
# For stub-labeled and vertex-labeled simple graphs, use graphspace = "SimpleStub" and "SimpleVertex", respectively.
# For stub-labeled and vertex-labeled loopy graphs, use graphspace = "LoopyOnlyStub" and "LoopyOnlyVertex", respectively.
# For stub-labeled and vertex-labeled loopy multigraphs, use graphspace = "MultiLoopyStub" and "MultiLoopyVertex", respectively.
# For stub-labeled and vertex-labeled multigraphs, use graphspace = "MultiOnlyStub" and "MultiOnlyVertex", respectively.
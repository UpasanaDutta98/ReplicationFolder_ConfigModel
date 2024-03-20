import networkx as nx
import copy
import numpy as np
import math
from scipy import stats
import dbl_edge_mcmc as mcmc
import multiprocessing
import pandas as pd
import time
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

def autocorrelation_function(series, alpha):
    n = len(series)
    data = np.asarray(series)
    xbar = np.mean(data)
    c0 = np.sum((data - xbar) ** 2)
    
    def standard_autocorrelations(h):
        corr = ((data[: n - h] - xbar) * (data[h:] - xbar)).sum() / c0
        mean = -(n-h)/(n*(n-1))
        var = (n**4 - (h + 3)*n**3 + 3*h*n*n + 2*h*(h+1)*n - 4*h*h)/((n+1)*n*n*(n-1)**2)
        SE = math.sqrt(var)
        standard_corr = (corr - mean)/SE
        return standard_corr

    y = standard_autocorrelations(1) # h = lag = 1
        
    z_critical = stats.norm.ppf(1 - alpha) # One-sided test

    return y, z_critical

def progress_D_chains(step_function, T, gap, D, swaps_so_far, last_r, n, m, last_graph, allow_loops, allow_multi, rejected, G_degree, denominator, increment, r_datapoints):
    num_swaps_needed = T*gap - 1
    for i in range(D):
        swaps = swaps_so_far
        counter = 0
        new_r = last_r[i]
        A = np.zeros(shape=(n,n))
        convert_edgelist_to_AdjMatrix(last_graph[i][0], A)              
        while swaps < num_swaps_needed:
            rejected[0] = 0                    
            step_function(A, last_graph[i][0], last_graph[i][1], rejected, allow_loops, allow_multi)
            delta_r = 0
            new_swap_list = last_graph[i][1]
            if rejected[0] != 1: # A swap was performed
                numerator = G_degree[new_swap_list[0]][1]*G_degree[new_swap_list[2]][1] + G_degree[new_swap_list[1]][1]*G_degree[new_swap_list[3]][1] - G_degree[new_swap_list[0]][1]*G_degree[new_swap_list[1]][1] - G_degree[new_swap_list[2]][1]*G_degree[new_swap_list[3]][1]
                delta_r = 2*numerator*2*m/denominator
            new_r = new_r + delta_r

            if counter%increment == 0:
                r_datapoints[i].append(new_r)

            swaps += 1
            counter+= 1
        last_r[i] = new_r    

def get_num_sig_autocorrelations(D, T, r_datapoints, gap, increment, alpha):
    sig = 0
    for i in range(D):
        List_of_r = []
        j = 0
        for k in range(T):
            List_of_r.append(r_datapoints[i][j])
            j += (gap//increment)
        autocorrelation_returned = autocorrelation_function(List_of_r, alpha)
        Rh_value = autocorrelation_returned[0]
        critical_value = autocorrelation_returned[1]
        if Rh_value > critical_value:
            sig += 1
    return sig


def get_samplingGap(G, graphname, increment, allow_loops, allow_multi, is_vertex_labeled, set_base_1 = -1, T = 500, alpha = 0.04, D = 10, upper_bound = 1):   
    total_time = 0
    flag = 0
    G_degree = list(nx.degree(G))
    m = G.number_of_edges()
    n = G.number_of_nodes()   
    
    S1 = 2*m
    S2 = 0
    S3 = 0
    for i in range(n):
        S2 += (G_degree[i][1])**2
        S3 += (G_degree[i][1])**3

    denominator = S1*S3 - (S2**2) # We calculate the denominator of r only once.


    if allow_multi == False:
        base_sampling_gap = int(1*m)
    else:
        if is_vertex_labeled == True: 
            base_sampling_gap = int(1*m)
        else:
            base_sampling_gap = int(1*m)

    if set_base_1 == 1:
        base_sampling_gap = 1
    increment = int(0.05*m) # The gap is increased by 5% of m with every check.            
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
    for j in range(1000*m): # Burn-in period
        step_function(A, edge_list, swaps, rejected, allow_loops, allow_multi)

    SL = 0
    for e in edge_list:
        SL += 2*G_degree[e[0]][1]*G_degree[e[1]][1]
    numerator = S1*SL - (S2**2)
    r_burn_in = numerator/denominator

    eta0_dict = -1

    r_datapoints = [[] for i in range(D)]
    for i in range(D):    
        r_datapoints[i].append(r_burn_in)
    last_graph = [] 
    for i in range(D):
        last_graph.append([copy.deepcopy(edge_list), copy.deepcopy(swaps)])
    last_r = []
    for i in range(D):
        last_r.append(r_burn_in)       

    gap = base_sampling_gap
    starttime = time.time()
    time1 = time.time()
    swaps_so_far = 0
    
    while True:
        progress_D_chains(step_function, T, gap, D, swaps_so_far, last_r, n, m, last_graph, allow_loops, allow_multi, rejected, G_degree, denominator, increment, r_datapoints)
        swaps_so_far = T*gap - 1
        time2 = time.time()
        sig = get_num_sig_autocorrelations(D, T, r_datapoints, gap, increment, alpha)
        if sig <= upper_bound and gap == base_sampling_gap:
            flag = 1 # Just for sanity check
        if sig <= upper_bound:
            eta0 = gap
            break

        gap = gap + increment

    sampling_gap = eta0
    endtime = time.time()
    total_time = (endtime-starttime)/3600
    return sampling_gap, flag, total_time

def save_sampling_gap_data(G, graphname, increment, lock, graphspace):
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
    
    gap, flag, total_time= get_samplingGap(G, graphname, increment, allow_loops=allow_loops, allow_multi=allow_multi, is_vertex_labeled=is_vertex_labeled)
    # lock before the process enters here
    lock.acquire()
    print("Network = ", graphname, G.number_of_nodes(), G.number_of_edges())
    print("Sampling Gap = ", gap)
    
    # Create the subfolder in Output folder if it does not exist already.
    folderpath = "../../Output/EstimatedSamplingGaps/"
    if not os.path.isdir(folderpath):
        os.makedirs(folderpath)

    # If dataframe not already present in the folder, create it (one-time)
    try:
        df = pd.read_csv("../../Output/EstimatedSamplingGaps/" + graphspace + "_SamplingGaps.csv")
    except:
        df = pd.DataFrame(columns=['Network', 'nodes', 'edges', 'gap', 'flag', 'time'])
        df.to_csv("../../Output/EstimatedSamplingGaps/" + graphspace + "_SamplingGaps.csv", index=False)  
    
    networks = list(df['Network'])
    networks.append(graphname)
    nodes = list(df['nodes'])
    nodes.append(G.number_of_nodes())
    edges = list(df['edges'])
    edges.append(G.number_of_edges())
    sampling_gaps = list(df['gap'])
    sampling_gaps.append(gap)
    flags = list(df['flag'])
    flags.append(flag)
    times = list(df['time'])
    times.append(total_time)
    
    New_df = pd.DataFrame()
    New_df['Network'] = networks
    New_df['nodes'] = nodes
    New_df['edges'] = edges
    New_df['gap'] = sampling_gaps
    New_df['flag'] = flags
    New_df['time'] = times
    
    New_df.to_csv("../../Output/EstimatedSamplingGaps/" + graphspace + "_SamplingGaps.csv", index=False)
    lock.release()
    # release lock here

def store_sampling_gaps(list_of_graphs, list_of_names, list_of_increments, graphspace):
    manager = multiprocessing.Manager()
    lock = multiprocessing.Lock() # Using multi-processing to process multiple networks parallely.
    jobs = []
    for i in range(len(list_of_graphs)):
        G = list_of_graphs[i]
        graphname = list_of_names[i]
        increment = list_of_increments[i]
        #save_sampling_gap_data(G, graphname, increment, lock, graphspace)
        p = multiprocessing.Process(target=save_sampling_gap_data, args=(G, graphname, increment, lock, graphspace))
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
    list_of_increments = []
    
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
        list_of_increments.append(m//20) #increment = 5% of m
            
        list_of_graphs.append(G3)
        list_of_names.append(str(textfileName[:-4]))
        
    store_sampling_gaps(list_of_graphs, list_of_names, list_of_increments, graphspace)

if __name__ ==  '__main__':
    graphspace = sys.argv[1] #"SimpleVertex"
    run(graphspace = graphspace)
# For stub-labeled and vertex-labeled simple graphs, use graphspace = "SimpleStub" and "SimpleVertex", respectively.
# For stub-labeled and vertex-labeled loopy graphs, use graphspace = "LoopyOnlyStub" and "LoopyOnlyVertex", respectively.
# For stub-labeled and vertex-labeled loopy multigraphs, use graphspace = "MultiLoopyStub" and "MultiLoopyVertex", respectively.
# For stub-labeled and vertex-labeled multigraphs, use graphspace = "MultiOnlyStub" and "MultiOnlyVertex", respectively.
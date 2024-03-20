## Instructions

The numbers at the start of each file name denotes the order in which the file should be run. File names starting with the same numbers can be run in any order w.r.t each other. The python scripts should be run before the notebooks, to generate the data needed to make the plots.

1. For Figure 4, generate the sampling gap estimates by running `0_Run_EstimatedSamplingGap.py` and then make the plots using the notebook 0_EstimatedSamplingGaps_Plot.ipynb.
* The script `0_Run_EstimatedSamplingGap.py` should be run individually for each of the 8 graph-space: SimpleVertex, SimpleStub, LoopyOnlyStub, LoopyOnlyVertex, MultiLoopyStub, MultiLoopyVertex, MultiOnlyStub, MultiOnlyVertex. Pass the name of the graph space as a command line argument. As an example, for "SimpleVertex", pass it as a command line argument: `python 0_Run_EstimatedSamplingGap.py SimpleVertex`. These runs are computationally intensive, so they may take several hours. Parallel processing has been enabled.
* The results of the run would be stored in the [Output/EstimatedSamplingGaps]() folder.
* Once the runs for all eight graph spaces are complete, run the `0_EstimatedSamplingGaps_Plot.ipynb` notebook to generate Fig. 4.

2. Run the `0_Store_1000mGraphs.py` script for each of the eight graph spaces to store the networks obtained after applying _1000m_ swaps to the empirical networks. 
* The graph space name can be passed as a command line argument.
* The results of the run would be stored in the [Output/1000mNetworks]() and [Output/1000mNetworkStatistics]() folders.

3. For Figure 7, run the `1_NetworksDetected_SwapsDone.py` script for all eight graph spaces. 
* Note that the execution of `0_Run_EstimatedSamplingGap.py` and `0_Store_1000mGraphs.py` scripts as well as the `0_EstimatedSamplingGaps_Plot.ipynb` notebook is necessary before running this script.
* Pass the graph space as a command line argument as shown before.
* The [Output/NetworksDetectedAtConvergence]() folder will get populated with the results.
* Run the `1_AverageSwaps_Plot.ipynb` notebook after the script's execution, to generate Figure 7.


4. For Figure 6(b), run the `2_All8NetworkStatistics_DFGLS.py` script passing each graph-space as the command line argument as shown before.
* The `1_NetworksDetected_SwapsDone.py` script should be run before running this script.
* The [Output/All8NetworkStatisticsAtConvergence]() folder will get populated with the results.
* Run the `1_RateOfEarlyConvergenceDFGLS_Plot.ipynb` notebook to generate Figure 6b.
## Instructions

1. For Figure 10, run the `SelfloopMultiedgeSwapRejections.py` script.
* It does not take any command line arguments. It does not depend on the runs of any of the other scripts.
* The outputs would be stored in the [Output/SwapsRejectionRateForSimpleNetworks]() folder.
* Run the `RelativeRejectionRate.ipynb` notebook to generate Figure 10.

2. For Figure 11, run the `1_GelmanRubin.py`, `1_Geweke.py`, `1_RafteryLewis.py` scripts, each of which should be run individually for the eight graph spaces: SimpleVertex, SimpleStub, LoopyOnlyStub, LoopyOnlyVertex, MultiLoopyStub, MultiLoopyVertex, MultiOnlyStub, MultiOnlyVertex. Pass the name of the graph space as a command line argument. For example, for the "SimpleVertex" graph, one of the runs would be: `python 1_GelmanRubin.py SimpleVertex`
* The scripts `0_1_Run_EstimatedSamplingGap.py` and `0_2_Store_1000mGraphs.py` need to be run before running this script.
* Figure 11 can be produced by then running the `1_ComparisonWithOtherConvergenceDiagnostics.ipynb` notebook.

3. For Figure 13, one needs to run only the `1_All8NetworkStatistics.ipynb` notebook.

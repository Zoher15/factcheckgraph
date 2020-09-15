# FactCheckGraph
This project aims to create knowledge graphs from the media for automated fact-checking of new headlines and claims. The data that the KG is constructed from, consists of real-world claims, annotated by fact-checkers as True or False. We calculate a truth score using this KG for new headlines and/or claims by using using different network centrality measures. 

## Graph Creation
The graph creation is done by using a tool called FRED, which converts sentences to semantic graphs. Graph Creation for predicting the truth score for a claim is done by leaving the said claim out of graph creation.

## Graph Processing
Graph processing involves calculating the statistics of the graphs and finding intersects between graphs.

## Graph Analysis
Graph analysis involves mining shortest paths using different centrality metrics on the graphs for predicting the truth score of an unseen claim (unseen refers to a claim left out during the graph creation process).

## Graph Plotting
This plots the results of the scores collected from the graph analysis step using an ROC and the distribution using kernel density estimator with histograms.

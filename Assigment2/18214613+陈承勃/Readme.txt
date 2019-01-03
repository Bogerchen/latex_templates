1. The solutions to Assignment 2 is written in "Solutions.pdf".
2. The MATLAB codes are in foler "Codes".
3. To train the dataset, type "main(datapath, K, randomstate, Max_iteration, eps)" in the command line on MATLAB.
datapath: string, the path to load .csv file,
K: int, number of clusters,
randomstate: int, setting for random state,
Max_iteration: int, number of iterations
eps: double, the threshold to control early stop.

An example:
main('TrainingData_GMM.csv', 4, 123, 40, 1e-4)
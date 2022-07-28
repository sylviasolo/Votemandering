# Votemandering
Contains two notebooks (one for grid instances and the other for state maps), and a few data-files accompanying the codes.

This notebook implements our votemandering algorithm from our paper to find the optimal resource allocation and a target map that achieve the highest number of seats in two rounds, combined. It can be used to run large randomly generated samples with same specifications, for the purpose of reporting results with more confidence.

We use the process of Recombination to form a list of district maps within which we operate. We have generated the data files already, using the gerrychain package. We use this pool of maps and find our target map as the best amongst those.

We first get a list of potential target maps, then arrange those in decreasing order of round 2 wins. We then run our algorithm step by updating one target map entry at a time until we reach our optimal solution.

Inputs: (look at the input cell)

data files containing lists that represent map assignments (a large pool of maps).
budgets for both the parties
voter data (can be generated randomly within the code, given specifications)
a fairness metric bound
k: number of times we want to run instances of votemandering algorithm (number of samples)(instances are generated randomly given above inputs)
Output: For each sample, the following can also be obtained through printing and/or visualization codes. (Outputs of running one instance of the votemandering algorithm)

Optimal target map specifications
optimal budget allocation
Round 1 and round 2 results
This notebook can be used to compare the effects of compactness factor: two datafiles should be submitted, named datafile1 and datafile2 in the input cell. We can also play with the budget range, alpha and the number of samples. Outputs excel sheets showing total additional seats won for k samples, given all inputs, datafiles.


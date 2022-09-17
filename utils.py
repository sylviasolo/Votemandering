import networkx as nx

import pandas as pd
import numpy as np
import scipy
import random
import matplotlib.pyplot as plt
from itertools import combinations, groupby
#from IPython.core.display import HTML
import matplotlib.patches as mpatches
import gurobipy as gp
from gurobipy import GRB
import time 


# Function that creates random populations V and vote shares P, Q
# if majority = True, P always corresponds to the majority party
def RandomPopulation(NumberOfUnits, minbound, maxbound, distmin, distmax, majority = True):
    if maxbound > minbound: #defining total voter population using V
      V = np.random.randint(minbound, maxbound, size=NumberOfUnits)
    else :
      V = np.repeat(maxbound, NumberOfUnits)
    #P0 = np.random.rand(NumberOfUnits) 
    P0 = np.random.uniform(distmin, distmax, NumberOfUnits) #uniformly generate vote shares
    #P0 = np.random.choice([0.4,0.6], size=NumberOfUnits)
    Q0 = 1-P0 #minority party share
    if majority == True:
      if np.sum(np.multiply(V,Q0))> np.sum(np.multiply(V,P0)):
        P0, Q0 = Q0, P0
    return V, P0, Q0

#this code can be used to create vote shares that follow some definite given pattern
def DefinedPopulation(votepattern, NumberOfUnits, minbound, maxbound, alpha, majority = True):
    if maxbound > minbound: 
      V = np.random.randint(minbound, maxbound, size=NumberOfUnits)
    else :
      V = np.repeat(maxbound, NumberOfUnits)
    P0 = []
    for i in range(0,NumberOfUnits):
        if votepattern[i]>0.5:
          P0.append(np.random.uniform(0.5, 0.7))
        else:
          P0.append(np.random.uniform(0.3, 0.5))
    P0 = np.array(P0)
    Q0 = 1-P0
    votesA = alpha*np.multiply(V,P0)
    votesB = alpha*np.multiply(V,Q0)
    return V, P0, Q0, votesA, votesB

def clusters(G, P0, n):
    votesdict = dict(zip(G.nodes(), P0))
    for i in range(n):
        random_node = random.sample(G.nodes, 1)
        neighbors=[j for j in G.neighbors(random_node[0])]
        avglist = [votesdict[j] for j in neighbors]
        avg = sum(avglist)/len(avglist)
        if avg >0.5:
            votesdict[random_node[0]] = np.random.uniform(0.6, 0.8)
        else:
            votesdict[random_node[0]] = np.random.uniform(0.2, 0.4)
    Pnew = list(votesdict.values())
    return Pnew
        
def moranI(G, P0):
    votesharedict = dict(zip(G.nodes(), P0))
    numerator = 0
    denominator = 0
    edgelist = [edge for edge in G.edges()]
    mean = sum(P0)/len(P0)
    for edge in edgelist:
        node1= edge[0]
        node2= edge[1]
        numerator = numerator + (votesharedict[node1]-mean)* (votesharedict[node2]-mean)
    for i in list(P0):     
        denominator = denominator + (i-mean)**2
    return len(P0)*numerator/(len(edgelist)*denominator)        


#Uniform allocation that party B (minority) uses
def PopUniformAllocationB(NumberOfUnits, BudgetB, V, votesB):
    matrix = V
    B = matrix*BudgetB/(sum(matrix))
    NewvotesB = np.zeros(NumberOfUnits)
    for i in range(NumberOfUnits):
        NewvotesB[i] = votesB[i] + B[i]
    return NewvotesB, B

def gridgraph(nodesx, nodesy):
    G = nx.grid_graph([nodesy,nodesx])
    #G = nx.grid_2d_graph(nodesy,nodesx) 
    dict1= dict(zip(G.nodes, range(nodesx*nodesy)))
    neighbors = []
    for i in G.nodes:
      neighbors.append([dict1[n] for n in G.neighbors(i)])
    return neighbors, G

def createlabels(Y,G):
    labels = {}
    i=0
    for node in G.nodes:
        if Y[i]>0.5:
            labels[node]='A'
        i=i+1
    return labels

def plotgraphs(G, NumberOfUnits, NumberOfDistricts, initmap, targetmap, NewvotesA, NewvotesB, votesA, votesB):
    #my_pos={x: x for x in G.nodes()},          
    my_pos = nx.spring_layout(G, seed = 500)
    my_pos = {(x,y):(y,-x) for x,y in G.nodes()}
    figsize=(10, 10)
    pos_higher = {}
    pos_lower = {}
    x_off = 0.3 # offset on the x axis
    y_off = 0.3 # offset on the y axis
    red_patch = mpatches.Patch(color='red', label='Votes for A')
    black_patch = mpatches.Patch(color='black', label='Votes for B')
    #black_patch = mpatches.Patch(color=colors[1], label='Votes for B')
    cmap=plt.cm.Spectral

    
    for k, v in my_pos.items():
        pos_higher[k] = (v[0]+x_off, v[1]+y_off)
    
    for k, v in my_pos.items():
        pos_lower[k] = (v[0]+x_off, v[1]-y_off)

#1 Initial map with original vote shares
    votesdiff1=[]
    W0=np.zeros((NumberOfUnits, NumberOfUnits))
    for i in range(0, NumberOfDistricts): 
      ind = np.where(initmap==i)
      votesdiff1.append(round(np.sum(votesA[ind])- np.sum(NewvotesB[ind])))
      if np.sum(votesA[ind])> np.sum(votesB[ind]):
        for j in ind:
          W0[ind[0],j] = i+1
    W0 = np.sum(W0, axis=0)

    '''
    plt.figure(6, figsize=(2,2))
    cmap=plt.cm.Spectral
    for i in range(len(votesdiff1)):
        plt.scatter([],[], c= [cmap(i/NumberOfDistricts)], label='VoteDiff {}'.format(votesdiff1[i]))

    plt.legend()
    '''

    plt.figure(1, figsize=figsize)
    plt.margins(x=0.25, y=0.25)
    ax = plt.gca()
    ax.set_title('Original map and original data: given data for investing')
    #G1 = nx.draw(G, node_color = [cmap(v/NumberOfDistricts) for v in initmap],  pos = my_pos, ax=ax)
    G1 = nx.draw(G, node_color =  initmap,  pos = my_pos, ax=ax)
    labels = createlabels(W0,G) 
    nx.draw_networkx_labels(G1, my_pos, labels, font_size=20, font_color='w')
    
    labels = {}
    i=0
    for node in G.nodes:
        labels[node]= str(round(votesA[i]))
        i=i+1
    nx.draw_networkx_labels(G1,pos_higher,labels,font_size=10,font_color='r')
    
    labels = {}
    i=0
    for node in G.nodes:
        labels[node]= str(round(votesB[i]))
        i=i+1
    nx.draw_networkx_labels(G1,pos_lower,labels,font_size=10,font_color='k')
    plt.legend(handles=[red_patch, black_patch],bbox_to_anchor=(1.05, 1))
  
    
#1.5 Showing A's investment on the original map, and the vote difference after B invests

    W05=np.zeros((NumberOfUnits, NumberOfUnits))
    for i in range(0, NumberOfDistricts): 
      ind = np.where(initmap==i)
      if np.sum(votesA[ind])> np.sum(votesB[ind]):
        for j in ind:
          W05[ind[0],j] = i+1
    W05 = np.sum(W05, axis=0)

  
    plt.figure(7, figsize=figsize)
    plt.margins(x=0.25, y=0.25)
    ax = plt.gca()
    ax.set_title('Budget investment by A and the vote differences after B invests: on the original map')
    G15 = nx.draw(G, node_color = [cmap(v/NumberOfDistricts) for v in initmap],  pos = my_pos, ax=ax)
    #G15 = nx.draw(G, node_color =  initmap,  pos = my_pos, ax=ax)
    labels = createlabels(W05,G) 
    nx.draw_networkx_labels(G15, my_pos, labels, font_size=20, font_color='w')
    
    labels = {}
    i=0
    for node in G.nodes:
        labels[node]= str(round(NewvotesA[i])-round(votesA[i]))
        i=i+1
    nx.draw_networkx_labels(G15,pos_higher,labels,font_size=10,font_color='r')
    
    cmap=plt.cm.Spectral
    for i in range(len(votesdiff1)):
        plt.scatter([],[], c= [cmap(i/NumberOfDistricts)], label='VoteDiff {}'.format(votesdiff1[i]))

    plt.legend()

#2 On the original map with updated vote shares by both parties: result of round 1
    votesdiff2=[]
    W2=np.zeros((NumberOfUnits, NumberOfUnits))
    for i in range(0, NumberOfDistricts): 
      ind = np.where(initmap==i)
      votesdiff2.append(np.sum(NewvotesA[ind])- np.sum(NewvotesB[ind]))
      if np.sum(NewvotesA[ind])> np.sum(NewvotesB[ind]):
        for j in ind:
          W2[ind[0],j] = i+1
    W2 = np.sum(W2, axis=0)

    plt.figure(2, figsize=figsize)
    plt.margins(x=0.25, y=0.25)
    ax = plt.gca()
    ax.set_title('On the original map after strategic campaigning: Rd 1 result')
    G2 = nx.draw(G, node_color= initmap,  pos = my_pos, ax=ax)
    labels = createlabels(W2,G) 
    nx.draw_networkx_labels(G2,my_pos,labels,font_size=20,font_color='w')

    
    labels = {}
    i=0
    for node in G.nodes:
        labels[node]= str(round(NewvotesA[i]))
        i=i+1
    nx.draw_networkx_labels(G2,pos_higher,labels,font_size=10,font_color='r')
   
    labels = {}
    i=0
    for node in G.nodes:
        labels[node]= str(round(NewvotesB[i]))
        i=i+1
    nx.draw_networkx_labels(G2,pos_lower,labels,font_size=10,font_color='k')
    plt.legend(handles=[red_patch, black_patch],bbox_to_anchor=(1.05, 1))

#2.5  A's investment on the target map and vote difference after B invests on the target map
    votesdiff25=[]
    W25=np.zeros((NumberOfUnits, NumberOfUnits))
    for i in range(0,NumberOfDistricts):
      ind = np.where(targetmap==i)
      votesdiff25.append(round(np.sum(votesA[ind]))- round(np.sum(NewvotesB[ind])))
      if np.sum(votesA[ind])> np.sum(NewvotesB[ind]):
        for j in ind:
          W25[ind[0],j] = i+1
    W25 = np.sum(W25, axis=0)

  
    plt.figure(8, figsize=figsize)
    plt.margins(x=0.25, y=0.25)
    ax = plt.gca()
    ax.set_title('Budget investment by A and vote differnces after B invests: on target map')
    G25 = nx.draw(G, node_color = [cmap(v/NumberOfDistricts) for v in targetmap],  pos = my_pos, ax=ax)
    #G15 = nx.draw(G, node_color =  initmap,  pos = my_pos, ax=ax)
    labels = createlabels(W25,G) 
    nx.draw_networkx_labels(G25, my_pos, labels, font_size=20, font_color='w')
    
    labels = {}
    i=0
    for node in G.nodes:
        labels[node]= str(round(NewvotesA[i])-round(votesA[i]))
        i=i+1
    nx.draw_networkx_labels(G25,pos_higher,labels,font_size=10,font_color='r')
    
    cmap=plt.cm.Spectral
    for i in range(len(votesdiff1)):
        plt.scatter([],[], c= [cmap(i/NumberOfDistricts)], label='VoteDiff {}'.format(votesdiff25[i]))

    plt.legend()

#3 The votemandered map that is shown to be fair using updated vote shares
    votesdiff3=[]
    W3=np.zeros((NumberOfUnits, NumberOfUnits))
    for i in range(0,NumberOfDistricts):
      ind = np.where(targetmap==i)
      votesdiff3.append(np.sum(NewvotesA[ind])- np.sum(NewvotesB[ind]))
      if np.sum(NewvotesA[ind])> np.sum(NewvotesB[ind]):
        for j in ind:
          W3[ind[0],j] = i+1
    W3 = np.sum(W3, axis=0)

    plt.figure(3, figsize=figsize)
    plt.margins(x=0.25, y=0.25)
    ax = plt.gca()
    ax.set_title('On the target map after strategic campaigning: should be fair')
    G3 = nx.draw(G, node_color= targetmap, pos = my_pos, ax=ax)
    labels = createlabels(W3,G) 
    nx.draw_networkx_labels(G3,my_pos,labels,font_size=20,font_color='w')

    labels = {}
    i=0
    for node in G.nodes:
        labels[node]= str(round(NewvotesA[i]))
        i=i+1
    nx.draw_networkx_labels(G3,pos_higher,labels,font_size=10,font_color='r')
   
    labels = {}
    i=0
    for node in G.nodes:
        labels[node]= str(round(NewvotesB[i]))
        i=i+1
    nx.draw_networkx_labels(G3,pos_lower,labels,font_size=10,font_color='k')
    plt.legend(handles=[red_patch, black_patch],bbox_to_anchor=(1.05, 1))


#4 The final target map with original vote shares: Rd 2 results
    votesdiff4=[]
    W1=np.zeros((NumberOfUnits, NumberOfUnits))
    for i in range(0,NumberOfDistricts):
      ind = np.where(targetmap==i)
      votesdiff4.append(np.sum(votesA[ind])- np.sum(votesB[ind]))
      if np.sum(votesA[ind])> np.sum(votesB[ind]):
        for j in ind:
          W1[ind[0],j] = i+1
    W1 = np.sum(W1, axis=0)

    plt.figure(4, figsize=figsize)
    plt.margins(x=0.25, y=0.25)
    ax = plt.gca()
    ax.set_title('On the target map with original vote data: max this 2')
    G2 = nx.draw(G, node_color= targetmap, pos = my_pos, ax=ax)
    labels = createlabels(W1,G) 
    nx.draw_networkx_labels(G2,my_pos,labels,font_size=20,font_color='w')

    
    labels = {}
    i=0
    for node in G.nodes:
        labels[node]= str(round(votesA[i]))
        i=i+1
    nx.draw_networkx_labels(G2,pos_higher,labels,font_size=10,font_color='r')
    
    labels = {}
    i=0
    for node in G.nodes:
        labels[node]= str(round(votesB[i]))
        i=i+1
    nx.draw_networkx_labels(G2,pos_lower,labels,font_size=10,font_color='k')
    plt.legend(handles=[red_patch, black_patch],bbox_to_anchor=(1.05, 1))
    return G1


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
import utils

#Function for computing A's maximum wins using voter data and budget    
def round1max(votesA, NewvotesB, initmap, budget, NumberOfDistricts):
    cols = len(initmap)
    D=[]
    wins = 0
    for i in range(0,NumberOfDistricts):
        ind = np.where(initmap==i)
        diff = np.sum(NewvotesB[ind])- np.sum(votesA[ind])
        if diff>0:
            D.append(diff)
        else:
            wins = wins+1
    D.sort()
    B = budget
    while B>0:
        if len(D)==0:
          break
        if B > D[0]:
          B = B- D[0]
          wins = wins+1
          D.pop(0)
        else:
          B = B- D[0]
    return wins

#Sets up a dataframe that has all the processed information about maps in the list
def InitialSetup(votesA, NewvotesB, votesB, dataframe0, initmap, budget, NumberOfDistricts):
    dataframe = dataframe0
    TotalVotes = sum(votesA)+ sum(votesB)
    rows, cols = dataframe.shape
    seats = []
    s = 0
    wastedvotes = []
    for j in range(0,rows):
      v1 = dataframe.iloc[j].to_numpy()
      s=0
      W = 0 
      for i in range(0,NumberOfDistricts):
        ind = np.where(v1==i)
        if np.sum(votesA[ind])> np.sum(votesB[ind]):
            s=s+1
            W = W + (3*np.sum(votesB[ind])-np.sum(votesA[ind]))/2
        else:
            W = W + (np.sum(votesB[ind])-3*np.sum(votesA[ind]))/2
      seats.append(s)
      wastedvotes.append(abs(W/TotalVotes))
    wins = round1max(votesA, NewvotesB, initmap, budget, NumberOfDistricts)
    dataframe['Max wins round 1']= wins #maximum wins in round 1 by party A (gets updated)
    dataframe['Wins in round 2'] = seats #wins in round 2 achieved by target maps
    
    #total wins = round 1+2, this column gets updated as we run our algorithm
    dataframe['Updated wins']= dataframe[['Max wins round 1', 'Wins in round 2']].sum(axis=1)
    dataframe['Fair?'] = 'Not yet' #this column gets updated as we run our algorithm
    dataframe['B-A Rd2 Egap'] = wastedvotes
    dataframe['FooledEgap'] = 0 #this column gets updated as we run our algorithm
    
    #sorting the list by the total number of wins, gets updated as we run our algorithm
    dataframe = dataframe.sort_values(by=['Updated wins'], ascending=False)
    
    return dataframe

def MyAlgoStep(votesA, NewvotesB, df, initmap, budget, NumberOfDistricts, Egap, alpha):
    BigM = budget*1000
    TotalVotes = sum(votesA)+ sum(NewvotesB) + budget
    NumberOfUnits = len(initmap)
    NoOfUnits = NumberOfUnits
    W=0
    v0 = df.iloc[0, 0:NumberOfUnits]
    #computing the efficiency gap (wasted votes) of the target map, after B invests.
    v1 = v0.to_numpy()
    for i in range(0, NumberOfDistricts):
        ind = np.where(v1==i)
        if np.sum(votesA[ind])> np.sum(NewvotesB[ind]):
            W = W + (3*np.sum(NewvotesB[ind])-np.sum(votesA[ind]))/2
        else:
            W = W + (np.sum(NewvotesB[ind])-3*np.sum(votesA[ind]))/2
    deficit = (3/2)*budget+ Egap*TotalVotes - W
    #print(W/TotalVotes)
    #check the basic constraint of whether we have enough budget to bring the number of 
    # wasted votes within the allowed range. If not, mark `infeasible' and return immediately.
    if deficit < 0:
      df.iloc[0,NumberOfUnits+3] = 'Infeasible'
      df.iloc[0,NumberOfUnits+5] = 50
      df.iloc[0,NumberOfUnits]= 0
      Total2rounds= df.iloc[0,NumberOfUnits]+ df.iloc[0,NumberOfUnits+1]
      df.iloc[0,NumberOfUnits+2]= Total2rounds
      df = df.sort_values(by=['Updated wins'], ascending=False)
      stop = 'no'
      #print('No need to check')
      return df, stop, 0, 0, 0, 0
    else:
      #print('Need to check')
    
      # If not infeasible, we create the optimization model and find the number of rd 1 wins.

      # Creation of a Concrete Model
      model = gp.Model()
      model.Params.LogToConsole = 0

      # indexing for defining variables
      units = range(len(initmap))  #preparing a set
      districts = range(NumberOfDistricts)

      # declare decision variables
      X = model.addVars(districts,  vtype='B') #indicating round 1 wins
      Y = model.addVars(districts,  vtype='B')  #indicating round 2 wins
      T = model.addVars(districts,   vtype='C', lb= -GRB.INFINITY) #diff between wasted votes in i'th district
      b = model.addVars(units,  vtype='C', lb= 0) #budget spent in each unit 
      v = model.addVars(units,  vtype='C', lb= 0) #Newvotes for A in each unit 
      
    
      #Objective function
      totalseats = model.setObjective( 
          expr = sum(X[i] for i in districts), sense=gp.GRB.MAXIMIZE)

      #building variables that we need in the constraints
      targetmap = df.iloc[0].to_numpy()
      targetmap = targetmap[0:NumberOfUnits]
      setI = []
      for i in range(0,NumberOfDistricts):
          ind = np.where(initmap==i)
          setI.append(ind[0])


      setJ = []
      for i in range(0,NumberOfDistricts):
          ind = np.where(targetmap==i)
          setJ.append(ind[0])

      #Constraints : budget constraints
      
      for i in units:
          model.addConstr(
          votesA[i]+b[i] == v[i], name="budgetconstraints")
    
      for i in units:
          model.addConstr(
          b[i] <=  votesA[i]*(1/alpha-1), name="budgetconstraints2")
          
      model.addConstr(sum(b[i] for i in units) <= budget,  name="totalbudgetconstraint")
          
      #Constraints : Round 1 constraints
      
      
      for i in districts:
          model.addConstr(
          sum(v[k] for k in setI[i])- sum(NewvotesB[k] for k in setI[i])>=
          1 - BigM*(1-X[i]), name="Rd1constraints1")

         
      for i in districts:
          model.addConstr(
          sum(v[k] for k in setI[i])- sum(NewvotesB[k] for k in setI[i])<=
            BigM*(X[i]), name="Rd1constraints2")
      
      #Constraints : Round 2 constraints
      
      for i in districts:
          model.addConstr(
          sum(v[k] for k in setJ[i])- sum(NewvotesB[k] for k in setJ[i])>=
          1 - BigM*(1-Y[i]), name="Rd2constraints1")

      
      for i in districts:
          model.addConstr( sum(v[k] for k in setJ[i])- 
                          sum(NewvotesB[k] for k in setJ[i])<= BigM*(Y[i]), name="Rd2constraints2")

      #Constraints : Round 2 wasted votes
      
      for i in districts:
          model.addConstr( - T[i] + (3*sum(NewvotesB[k] for k in setJ[i])- 
          sum(v[k] for k in setJ[i]))/2 <= BigM*(1-Y[i]), name="Tconstraints1")
 
      
      for i in districts:
          model.addConstr(- T[i] + (3*sum(NewvotesB[k] for k in setJ[i])- 
          sum(v[k] for k in setJ[i]))/2 >= 0, name="Tconstraints2")
      
      
      for i in districts:
          model.addConstr(T[i] - (sum(NewvotesB[k] for k in setJ[i])- 
          3*sum(v[k] for k in setJ[i]))/2 <= BigM*(Y[i]), name="Tconstraints3")

      
      for i in districts:
          model.addConstr( T[i] - (sum(NewvotesB[k] for k in setJ[i])- 
          3*sum(v[k] for k in setJ[i]))/2 >= 0, name="Tconstraints4")
      
      
      model.addConstr((sum(T[i] for i in districts))/TotalVotes <= Egap,name="Egapcon") 
      #model.addConstr((sum(T[i] for i in districts))/TotalVotes >= -Egap) 
      model.update()

      #Solution 
    
      model.optimize()
      '''
      model.computeIIS()
      
      model.write("model.ilp")
      if model.status == GRB.INFEASIBLE:
        model.feasRelaxS(1, False, False , True)
        model.optimize()
      '''
      
      WastedVotes = []
      if model.status == GRB.INFEASIBLE:
          total = 0
          df.iloc[0,NoOfUnits+3] = 'Infeasible'
      else:
          total = 0
          df.iloc[0,NoOfUnits+3] = 'Yes'
          for i in range(NumberOfDistricts):
            total = total + X[i].getAttr(GRB.Attr.X)
            WastedVotes.append(T[i].getAttr(GRB.Attr.X))
      
      if model.status == GRB.INFEASIBLE:
          EgapTotal = 50
          df.iloc[0,NoOfUnits+5] = EgapTotal
      else:    
          EgapTotal = sum(WastedVotes)/TotalVotes
          df.iloc[0,NoOfUnits+5] = EgapTotal 
      
    
      df.iloc[0,NoOfUnits]= total
      Total2rounds= df.iloc[0,NoOfUnits]+ df.iloc[0,NoOfUnits+1]
      df.iloc[0,NoOfUnits+2]= Total2rounds
      df = df.sort_values(by=['Updated wins'], ascending=False)
      
      if df.iloc[0,NoOfUnits+2] > Total2rounds:
        stop = 'no'
      else: 
        #stop = 'no'
        stop = 'yes'

      BudgetAlloc = []
      NewvotesA = []
        
      if model.status == GRB.INFEASIBLE:
          return df, stop, BudgetAlloc, NewvotesA, WastedVotes, EgapTotal
      else:
        for i in range(NoOfUnits):
            bs = b[i].getAttr(GRB.Attr.X)
            vs = v[i].getAttr(GRB.Attr.X)
            BudgetAlloc.append(bs)
            NewvotesA.append(vs)
        return df, stop, BudgetAlloc, NewvotesA, WastedVotes, EgapTotal

def command(votesA, NewvotesB, votesB, df0, initmap, budgetA, Egap, alpha, maxiter, NumberOfDistricts):
    NumberOfUnits = len(initmap)
    df0 = df0.sample(frac=1)
    df0 = df0[:maxiter]
   
    df = InitialSetup(votesA, NewvotesB, votesB, df0, initmap, budgetA, NumberOfDistricts)
    
    stop = 'no'
    iter = 0
    while stop == 'no':
        df, stop, BudgetAlloc, NewvotesA, wastedvotes, EgapTotal = MyAlgoStep(votesA, NewvotesB, df, initmap, budgetA, NumberOfDistricts, Egap, alpha)
        iter = iter+1
        if iter > maxiter-1:
          print('All are infeasible')
          break

    targetmap = df.iloc[0].to_numpy()
    EgapRd2 = df.iloc[0,NumberOfUnits+4]

    return  targetmap, df, BudgetAlloc, NewvotesA, EgapRd2, EgapTotal


def analysis(sr1, df, BudgetAlloc, NewvotesA, NumberOfDistricts):
    #print(df['Fair?'].value_counts()) 
    #Newvotes = BudgetAlloc+votesA
    NumberOfUnits = len(NewvotesA)
    NewvotesA =  [round(x,3) for x in NewvotesA]
    NewvotesA = np.array( NewvotesA )
    BudgetAlloc =  [round(x,3) for x in BudgetAlloc]
    #print(EgapRd2) #of the target map using updated vote shares
    target = df.iloc[0].to_numpy()
    targetmap = target[0:NumberOfUnits]
    Max = NumberOfDistricts*2
    MaxNOstrat = sr1*2
    MaxStrat = df.iloc[0,NumberOfUnits+2]
    rd1wins=  df.iloc[0,NumberOfUnits]
    rd2wins=  df.iloc[0,NumberOfUnits+1]
    targetmapnumber = df.index[0]
    #print('Maximum possible wins in 2 rounds:', Max)
    #print('Wins without strategic campaigning:', MaxNOstrat)
    #print('Wins with strategic campaigning:', MaxStrat)
    #print(df[:5])
    return targetmap, BudgetAlloc, NewvotesA, Max, MaxNOstrat, MaxStrat, rd1wins, rd2wins, targetmapnumber


def Doeverything(datafile, initmapnumber, V, votesA, votesB, Egap, budgetA, budgetB, alpha, maxiter, NumberOfDistricts):
    df0 = pd.read_csv(datafile, index_col=False, header=None)
    
    
    initmap = df0.iloc[initmapnumber]
    
  
    NumberOfUnits = int(len(initmap))
    
    sr1=0 #number of seats in round 1 without strategic campaign
    for i in range(0,NumberOfDistricts):
      ind = np.where(initmap==i)
      if np.sum(votesA[ind])> np.sum(votesB[ind]):
          sr1=sr1+1


    NewvotesB, B = utils.PopUniformAllocationB(NumberOfUnits, budgetB, V, votesB)

    targetmap, dfupdated, BudgetAlloc, NewvotesA, EgapRd2, EgapTotal = command(votesA, NewvotesB, votesB, df0, initmap, budgetA, Egap, alpha, maxiter, NumberOfDistricts)
    targetmap, BudgetAlloc, NewvotesA, Max, MaxNOstrat, MaxStrat, rd1wins, rd2wins, targetmapnumber = analysis(sr1, dfupdated, BudgetAlloc, NewvotesA, NumberOfDistricts)
    Fairseries = dfupdated['Fair?'].value_counts()
    
    if len(Fairseries)<3:
      InfNumber = 0
      YesNumber = Fairseries[1]
    else: 
      InfNumber = Fairseries[1]
      YesNumber = Fairseries[2]
    #print(BudgetAlloc)
    return EgapTotal, initmap, dfupdated, targetmap, Max, MaxNOstrat, MaxStrat, rd1wins, rd2wins, EgapRd2, InfNumber, YesNumber, targetmapnumber, NewvotesA, NewvotesB
#print(sum(votesA), sum(votesB))

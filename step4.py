# -*- coding: utf-8 -*-

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from scipy.sparse import csr_matrix
from joblib import Parallel, delayed, cpu_count

class MIRCOfit:


    def __init__(self, CorR = 'C'):
        
        self.rules = dict()
        self.CorR = CorR # 'C' for classification, 'R' for regression
        self.numOfMissed = 0
        self.missedXvals = []
        self.initNumOfRules = 0
        self.numOfRules = 0


    def predict(self, xvals):
        
        # Parallel prediction
        p = cpu_count()
        xsets = np.array_split(xvals, p)
        
        chunkPreds = Parallel(n_jobs=p, prefer="threads")(
            delayed(self.chunkPredict)(x0) for x0 in xsets)
        
        if (self.CorR == 'C'):
            predictions = np.empty(shape=(0), dtype=int)
        else:
            predictions = np.empty(shape=(0), dtype=float)
        
        for indx in range(len(chunkPreds)):
            predictions = np.append(predictions, chunkPreds[indx]['predictions'])
            for x0 in chunkPreds[indx]['missedXvals']:
                self.missedXvals.append(x0)
            self.numOfMissed += chunkPreds[indx]['numOfMissed']

        # DEBUG:
        # if (self.numOfMissed > 0):
        #     print('Warning...')
        #     print('Total number of missed points:' + str(self.numOfMissed))

        return predictions        


    def chunkPredict(self, xvals):
        
        chunkPreds = dict()
        if (self.CorR == 'C'):
            chunkPreds['predictions'] = np.zeros(len(xvals), dtype=int)
        else:
            chunkPreds['predictions'] = np.zeros(len(xvals), dtype=float)
        chunkPreds['numOfMissed'] = 0
        chunkPreds['missedXvals'] = []
        
        for sindx, x0 in enumerate(xvals):
            totvals = np.zeros(len(self.rules[0][-1]), dtype=float)
            approxvals = np.zeros(len(self.rules[0][-1]), dtype=float)
            totnum, approxnum = 0, 0
            trueratios = np.zeros(len(self.rules))
            for rindx, rule in enumerate(self.rules.values()):
                truecount = 0
                # The last value in the list stands for
                # the numbers in each class
                for clause in rule[:-1]:
                    if (clause[1] == 'l'):
                        if (x0[clause[0]] <= clause[2]):
                            truecount = truecount + 1
                    if (clause[1] == 'r'):
                        if (x0[clause[0]] > clause[2]):
                            truecount = truecount + 1
                # Not the last one (class numbers)
                trueratios[rindx] = truecount/(len(rule)-1)
                if (trueratios[rindx] == 1.0):
                    totvals += rule[-1]
                    totnum += 1
                else:
                    approxvals += trueratios[rindx]*rule[-1]
                    approxnum += 1
                    
            # TODO: We may return the prediction probabilities
            if (sum(totvals) > 0.0):
                if (self.CorR == 'C'):
                    chunkPreds['predictions'][sindx] = np.argmax(totvals)
                else:
                    chunkPreds['predictions'][sindx] = (1.0/totnum)*totvals
            else:
                if (self.CorR == 'C'):
                    chunkPreds['predictions'][sindx] = np.argmax(approxvals)
                else:
                    chunkPreds['predictions'][sindx] = (1.0/approxnum)*approxvals
                
                chunkPreds['missedXvals'].append(x0)
                chunkPreds['numOfMissed'] += 1
                  
        return chunkPreds
    
    def exportRules(self):
        
        for rindx, rule in enumerate(self.rules.values()):
            print('RULE %d:' % rindx)
            # Last compenent stores the numbers in each class
            for clause in rule[:-1]:
                if (clause[1] == 'l'):
                    print('==> x[%d] <= %.2f' % (clause[0], clause[2]))
                if (clause[1] == 'r'):
                    print('==> x[%d] > %.2f' % (clause[0], clause[2]))

            strarray = '['
            for cn in rule[-1][0:-1]:
                strarray += ('{0:.2f}'.format(cn) + ', ')
            strarray += ('{0:.2f}'.format(rule[-1][-1]) + ']')
                
            print('==> Class numbers: %s' % strarray)        



    def exportRulesSimplified1(self):
        
       for rindx, rule in enumerate(self.rules.values()):
            print('RULE %d:' % rindx)
            rules_all = []
            rules_attribute = []
            rules_simplified = []
            for clause in rule[:-1]:
                rules_attribute.append(clause[0])  # Make a list of all the feature variables in a clause. This will be used to evaluate which clauses involve the same feature variable
                rules_all.append(clause) # Make a list of all the clauses for indexing purposes
            for rule_index in rules_attribute:
                # The following runs only if there is more than one clause which involves the same feature variable
                if rules_attribute.count(rule_index) > 1:
                    indices = [i for i, x in enumerate(rules_attribute) if x == rule_index] # Make a list of indices of the rules which involve the same feature variable
                    grouped = []
                    for index in indices:
                        grouped.append(rules_all[index]) # Make a list of the rules which involve the same feature variable. This will be used to evaluate the (in)equality signs
                        group_signs = []
                    for group in range(len(grouped)):
                        group_signs.append(grouped[group][1]) # Make a list of all the (in)equality signs of the rules which involve the same variable
                    index_group_l = [i for i, x in enumerate(group_signs) if x == 'l'] # Make a list of the indices of the rules which involve the same variable and which have the sign <=
                    index_group_r = [i for i, x in enumerate(group_signs) if x == 'r'] # Make a list of the indices of the rules which involve the same variable and which have the sign >
                    values_group_l = []
                    values_group_r = []
                    if len(index_group_l) != 0: 
                        for i in index_group_l:
                            values_group_l.append(grouped[i][2]) # Make a list of the values of the rules which involve the same variable and have the sign <=
                    if len(index_group_r) != 0:
                        for j in index_group_r:
                            values_group_r.append(grouped[j][2]) # Make a list of the values of the rules which involve the same variable and have the sign >
            
                    if ((len(index_group_l) == 0) and (len(index_group_r) != 0)):
                        rules_simplified.append([rule_index, 'r', max(values_group_r)]) # If all clauses which involve the same feature variable have the sign >, you take the max of the values
                
                    if ((len(index_group_l) != 0) and (len(index_group_r) == 0)):
                        rules_simplified.append([rule_index, 'l', min(values_group_l)]) # If all clauses which involve the same feature variable have the sign <=, you take the min of the values
                
                    if ((len(index_group_l) != 0) and (len(index_group_r) != 0)):
                        rules_simplified.append([rule_index, 'm', max(values_group_r), min(values_group_l)]) # If the clauses which involve the same feature variable have varying signs, you make a two-sided comparison 
                
                # If there is only one clause which involves a certain feature variable, then no change is made to the clause
                elif rules_attribute.count(rule_index) == 1:
                    index = rules_attribute.index(rule_index)
                    rules_simplified.append(rules_all[index])
            
                # This is done in order to make sure there are no duplicate rules when simplifying rules
                rules_simplified_final = []
                for i in rules_simplified:
                    if i not in rules_simplified_final:
                        rules_simplified_final.append(i)
                   
            for clause in rules_simplified_final:
                if (clause[1] == 'l'):
                    print('==> x[%d] <= %.2f' % (clause[0], clause[2]))
                if (clause[1] == 'r'):
                    print('==> x[%d] > %.2f' % (clause[0], clause[2]))
                if (clause[1] == 'm'):
                    print('==> %.2f < x[%d] <= %.2f' % (clause[2], clause[0], clause[3]))

            strarray = '['
            for cn in rule[-1][0:-1]:
                strarray += ('{0:.2f}'.format(cn) + ', ')
            strarray += ('{0:.2f}'.format(rule[-1][-1]) + ']')
            
            rules = []
            for clause in rule[:-1]:
                rules.append(clause)
                
            print('==> Class numbers: %s' % strarray)       
            
            
    def exportRulesSimplified2(self):
        
        # Getting the different rules  Contains all subrules
        l = 0  # number of subrules in total
        h = 0 # number of rules
        counter = []
        clauseprog = []
        for rule in self.rules.values():  
            k = 0 # number of subrules in a rule
            h += 1
            for clause in rule[:-1]:
                clauseprog.append(clause)
                l += 1
                k += 1
            counter.append(k) # Array with number of clauses in each rule
        index = np.zeros((l,1))    #[0]*l  
        
        counter_final = []
        j=0
        for i in range(len(counter)):
            j+=counter[i]
            counter_final.append(j)
            
        # every subrule is given an index, when two subrules are the same they get the same number 
        for i in range(l):
            if (index[i] != 0):
                index[i] = i
        for i in range(l):
            for j in range(l):
                if (clauseprog[i] == clauseprog[j]):
                    index[j] = i
                    
        # In this part the indices are found for the rules which do have the most subrules in common            
        rulesindices = np.split(index, counter_final) 
        differences = []
        indicator = np.zeros((h,h))
        indicesrules = []
        for i in range(h):
            for j in range(h):
                differences[i,j] = len(rulesindices[i].difference(rulesindices[j]))
                if (differences[i,j] == 1):
                    indicator[i,j] = 1    
        for i in range(h):
            indicesrules[i,:] = np.where(indicator[:, i] == 1)
        
        #the results are printed
        r = 0
        for rindx, rule in enumerate(self.rules.values()):
            print('RULE %d INTERSECTION RULE %d:' % (rindx,indicesrules[rindx,1]))
            
            for clause in rule[:-1]:
                for i in range(counter[rindx]+1):
                    if (index[r] == index[(sum(counter[0:rindx])+i)]):
                        if (clause[1] == 'l'):
                            print('==> x[%d] <= %.2f' % (clause[0], clause[2]))
                        if (clause[1] == 'r'):
                            print('==> x[%d] > %.2f' % (clause[0], clause[2]))
            r = r + 1
            
            print('RULE %d adding:' % rindx)
            for clause in rule[:-1]:
                for i in range(counter[rindx]+1):
                    if (index[r] != index[(sum(counter[0:rindx])+i)]):
                        if (clause[1] == 'l'):
                            print('==> x[%d] <= %.2f' % (clause[0], clause[2]))
                        if (clause[1] == 'r'):
                            print('==> x[%d] > %.2f' % (clause[0], clause[2]))

            strarray = '['
            for cn in rule[-1][0:-1]:
                strarray += ('{0:.2f}'.format(cn) + ', ')
            strarray += ('{0:.2f}'.format(rule[-1][-1]) + ']')
                
            print('==> Class numbers: %s' % strarray)      
            
class MIRCO:


    def __init__(self, rf):
        
        # rf is a fitted Random Forest!        
        self.rf = rf
        self.estimator = None
        self.featureNames = None


    def getRule(self, fitTree, nodeid):

        left = fitTree.tree_.children_left
        right = fitTree.tree_.children_right
        threshold = fitTree.tree_.threshold
        featurenames = [self.featureNames[i] for i in fitTree.tree_.feature]
    
        def recurse(left, right, child, lineage=None):
            if lineage is None:
                lineage = [child]
            if child in left:
                parent = np.where(left == child)[0].item()
                split = 'l'
            else:
                parent = np.where(right == child)[0].item()
                split = 'r'
    
            # The first in the list shows the feature index
            lineage.append((fitTree.tree_.feature[parent], split,
                            threshold[parent], featurenames[parent]))
    
            if parent == 0:
                lineage.reverse()
                return lineage
            else:
                return recurse(left, right, parent, lineage)
    
        rule = recurse(left, right, nodeid)
        # Weighted values for each class in leaf comes from tree_
        # These are later filled with actual numbers instead of weights
        rule[-1] = fitTree.tree_.value[nodeid][0]
    
        return rule   


    def greedySCP(self, c, A):
        
        # TODO: Can be faster by using heaps
        
        # Mathematical model
        # minimize     c'x
        # subject to   Ax >= 1
        #              x in {0,1}
        # c: n x 1
        # A: m x n
        
        # number of rows and number of columns
        m, n = A.shape
        # set of rows (items)
        M = set(range(m))
        # set of columns (sets)
        N = set(range(n))
        
        R = M
        S = set()
        while (len(R) > 0):
            minratio = np.Inf
            for j in N.difference(S):
                # Sum of covered rows by column j
                denom = np.sum(A[list(R), j])
                if (denom == 0):
                    continue
                ratio = c[j]/denom
                if (ratio < minratio):
                    minratio = ratio
                    jstar = j
            column = A[:, jstar]
            Mjstar = set(np.where(column.toarray() == 1)[0])
            R = R.difference(Mjstar)
            S.add(jstar)
    
        listS = list(S)
        # Sort indices
        sindx = list(np.argsort(c[listS]))
        S = set()
        totrow = np.zeros((m, 1), dtype=np.int32)
        for i in sindx:
            S.add(listS[i])
            column = A[:, listS[i]]
            totrow = totrow + column
            if (np.sum(totrow > 0) >= m):
                break
    
        return S


    def solveSCP(self, c, A):
        
        # Number of rows and number of columns
        m, n = np.shape(A)
        
        S = self.greedySCP(c, A)
        S = np.array(list(S), dtype=np.long)
                
        return S


    def fit(self, X, y):
        
        if (isinstance(self.rf, RandomForestClassifier)):
            fittedMIRCO = MIRCOfit(CorR = 'C')
        else:
            fittedMIRCO = MIRCOfit(CorR = 'R')
        
        nOfSamples, nOfFeatures = np.shape(X)
        nOfClasses = int(max(y) + 1) # classes start with 0
        
        self.featureNames = ['x[' + str(indx) + ']'
                     for indx in range(nOfFeatures)]

        criterion = self.rf.criterion
        
        # Total number of rules
        nOfRules = 0
        for fitTree in self.rf.estimators_:
            nOfRules += fitTree.get_n_leaves()
        
        # Initial number of rules is stored
        fittedMIRCO.initNumOfRules = nOfRules
        
        # Parallel construction of SCP matrices
        p = cpu_count()
        estsets = np.array_split(self.rf.estimators_, p)
        
        retdicts = Parallel(n_jobs=p, prefer="threads")(
            delayed(self.chunkFit)(X, y, est, criterion, fittedMIRCO.CorR)
            for chunkNo, est in enumerate(estsets))
        
        c = np.empty(shape=(0), dtype=np.float)
        rows = np.empty(shape=(0), dtype=np.int32)
        cols = np.empty(shape=(0), dtype=np.int32)
        colTreeNos = np.empty(shape=(0), dtype=np.int32)
        colLeafNos = np.empty(shape=(0), dtype=np.int32)
        colChunkNos = np.empty(shape=(0), dtype=np.int32)
        colno = 0
        for chunkNo in range(len(estsets)):
            ncols = len(retdicts[chunkNo]['c'])
            c = np.hstack((c, retdicts[chunkNo]['c']))
            rows = np.hstack((rows, retdicts[chunkNo]['rows']))
            colTreeNos = np.hstack((colTreeNos, retdicts[chunkNo]['colTreeNos']))
            colLeafNos = np.hstack((colLeafNos, retdicts[chunkNo]['colLeafNos']))
            tempcols = colno + retdicts[chunkNo]['cols']
            cols = np.hstack((cols, tempcols))
            colChunkNos = np.hstack((colChunkNos, np.ones(ncols,
                                                        dtype=np.int8)*chunkNo))
            colno = cols[-1]+1
        

        data = np.ones(len(rows), dtype=np.int8)
        A = csr_matrix((data, (rows, cols)), dtype=np.int8)
                
        S = self.solveSCP(c, A)
        
        for indx, col in enumerate(S):
            chunkno = colChunkNos[col]
            treeno = colTreeNos[col]
            fitTree = estsets[chunkno][treeno]
            leafno = colLeafNos[col]
            rule = self.getRule(fitTree, leafno)
            fittedMIRCO.rules[indx] = rule
            
            # Filling the last element in 'rule'
            # with actual numbers in each class
            # not the weighted numbers - Though,
            # we do not use weights for MIRCO
            y_rules = fitTree.apply(X)
            covers = np.where(y_rules == leafno)
            leafyvals = y[covers]  # yvals of the samples in the leaf
            unique, counts = np.unique(leafyvals, return_counts=True)
            numsinclasses = np.zeros(nOfClasses)
            for ix, i in enumerate(unique):
                numsinclasses[int(i)] = counts[ix]
            fittedMIRCO.rules[indx][-1] = numsinclasses
            
        fittedMIRCO.numOfRules = len(S)
        
        return fittedMIRCO


    def chunkFit(self, X, y, estimators, criterion, CorR):
        
        numRules = 0
        for fitTree in estimators:
            numRules += fitTree.get_n_leaves()
        
        retdict = dict()
        
        retdict['c'] = np.zeros(numRules, dtype=np.float)
        retdict['rows'] = np.empty(shape=(0), dtype=np.int32)
        retdict['cols'] = np.empty(shape=(0), dtype=np.int32)

        retdict['colLeafNos'] = np.zeros(numRules, dtype=np.int32)
        retdict['colTreeNos'] = np.zeros(numRules, dtype=np.int32)
        
        col = 0
        for treeno, fitTree in enumerate(estimators):
            # Tells us which sample is in which leaf
            y_rules = fitTree.apply(X)
            for leafno in np.unique(y_rules):
                covers = np.where(y_rules == leafno)[0]
                retdict['rows'] = np.hstack((retdict['rows'], covers))
                retdict['cols'] = np.hstack((retdict['cols'], 
                                             np.ones(len(covers), dtype=np.int8)*col))                
                leafyvals = np.array(y[covers]) # y values of the samples in the leaf
                if (CorR == 'C'): # classification
                    unique, counts = np.unique(leafyvals, return_counts=True)
                    probs = counts/np.sum(counts)
                    # Currently it is just Gini and Entropy
                    # TODO: Add other criteria
                    if (criterion == 'gini'):
                        retdict['c'][col] = 1 + (1 - np.sum(probs**2)) # 1 + Gini
                    else:
                        retdict['c'][col] = 1 + (-np.dot(probs, np.log2(probs))) # 1 + Entropy
                else: # regression
                    # Currently it is just MSE
                    # TODO: Add other criteria
                    leafyavg = np.average(leafyvals)
                    mse = np.average(np.square(leafyavg - leafyvals))
                    if (criterion == 'mse'):
                        retdict['c'][col] = 1.0 + mse # 1 + MSE
                retdict['colLeafNos'][col] = leafno
                retdict['colTreeNos'][col] = treeno
                col += 1
                
        return retdict
# Forest-Based Binary Classification Study

For the MSc Econometrics & Management Science, specializing in Business Analytics & Quantitative Marketing [BAQM], at the Erasmus School of Economics [ESE] in Rotterdam, I have conducted a computational study regarding forest-based binary classification for the course "Machine Learning" in Python awarded with a 9.5. In particular, I compared decision trees, random forests, and the novel MIRCO method with anticipated enhanced performance and interpretability capabilities, on a German Credit Dataset. In particular, I propose an algorithm that makes MIRCO a classification algorithm. Furthermore, I examine efficient and scalable implementations of MIRCO by comparing a greedy algorithm, and a List-and-Remove [LAR] algorithm [Yang et al., 2015] used to implement it. Moreover, I present simplifications to the MIRCO output that enhances interpretability of the results, such as compacting rules concerning the same variables, or providing a compact overview of the structure of the resulting rules by graphing a tree of pairs of sets of rules that have the most feature space in common.

# Replication Files

This repository contains the following files for the first assignment:
- Forest_Based_Binary_Classification_Study.pdf: Presents descriptions and results of the computational study concerning binary classification using decision trees, random forests, MIRCO, and adapted MIRCO.
- step1.py: This code implements Random Forest (RF), MIRCO and Decision Tree (DT) algorithms on the given credit dataset, and reports (i) the average performance of all three methods, (ii) the average number of missed points by MIRCO, and (iii) the average number of rules used by all three methods in Python.
- step2.py: This code implements our proposed improvement on Chvatal’s greedy set covering heuristic used in MIRCO in Python. The definition of the function ’greedySCP’ is altered.
- step4.py: This code implements our proposed simplifications in reporting the outcome of the MIRCO algorithm in Python. The functions ’exportRulesSimplified1’ and ’exportRulesSimplified2’ are added as simplification. ’exportRulesSimplified1’ is the implementation of the simplification given in the instructions. ’exportRulesSimplified2’ is the implementation of the simplification mentioned above.

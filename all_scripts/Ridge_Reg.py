import numpy as np
from sklearn.metrics import r2_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from sklearn.linear_model import RidgeCV
from regressors import stats

#python3 Linear_Reg.py one_hot_encode.csv> model_descr.csv

#read in LFC dataframe
one_hot_data = pd.read_csv(sys.argv[1])
sample_name = sys.argv[1].replace('.txt','')
sample_name = sample_name.split('/')[-1]

#################  Regression Model  ###########################
LFC_y = one_hot_data['LFC']
Count_y = np.log10(one_hot_data['Count']+0.5)
X= one_hot_data.drop(["T_T","A_A","Coord","Count","Local Mean","LFC","ORF ID", "ORF Name"],axis=1) # one hot encoded nucl at every position except T A

#perform cross validation and train-test the models

LFC_results = RidgeCV(cv =10, alphas=[0.1]).fit(X, LFC_y)
LFC_R2=LFC_results.score(X,LFC_y)

Count_results = RidgeCV(cv =10, alphas=[0.1]).fit(X, Count_y)
Count_R2=LFC_results.score(X,Count_y)

LFC_pvals = stats.coef_pval(LFC_results,X,LFC_y)
Count_pvals = stats.coef_pval(Count_results,X,Count_y)

print("--------------------- LFC Regression ---------------------")
print("Params: "+str(LFC_results.get_params()))
print(stats.summary(LFC_results,X,LFC_y))

print("--------------------- Count Regression ---------------------")
print("Params: "+str(Count_results.get_params()))
print(stats.summary(Count_results,X,Count_y))
Models = pd.DataFrame(data={"LFC Coef":LFC_results.coef_,"LFC Pvals":LFC_pvals[1:],"Count Coef": Count_results.coef_,"Count Pvals":Count_pvals[1:]})


# LFC Plot
"""
#Plot the last train-test split 
fig, (ax1) = plt.subplots(1, sharex=True, sharey=True)
ax1.set_title("Predicted vs. Actual LFC")
ax1.scatter(LFC_y_test,LFC_y_pred,s=1,c='green',alpha=0.5)
ax1.set_xlabel('Actual')
ax1.set_ylabel('Predicted')
ax1.text(-5, 5, "R2: "+ str(sum(LFC_R2_list) / len(LFC_R2_list)), fontsize=10)
ax1.axhline(y=0, color='k')
ax1.axvline(x=0, color='k')
ax1.plot([-6,6], [-6,6], 'k-', alpha=0.75, zorder=1)
ax1.set_xlim(-6,6)
ax1.set_ylim(-6,6)
ax1.grid(zorder=0)
#plt.show()
"""
#Coefficients of the regression
C=[]
T=[]
G=[]
A=[]

for idx,col in enumerate(X.columns):
        if "C" in col: C.append(LFC_results.coef_[idx])
        if "T" in col: T.append(LFC_results.coef_[idx])
        if "G" in col: G.append(LFC_results.coef_[idx])
        if "A" in col: A.append(LFC_results.coef_[idx])

fig, ax = plt.subplots(figsize=(20,5))
x = np.arange(40)
bar_width = 0.2
b1 = ax.bar(x-bar_width/2 - bar_width, C,width=bar_width,label="C")
b2 = ax.bar(x-bar_width/2 , T, width=bar_width,label="T")
b3 = ax.bar(x + bar_width/2 , G, width=bar_width,label="G")
b4 = ax.bar(x + bar_width/2 + bar_width , A, width=bar_width,label="A")
ax.set_xticks(range(40))
ax.set_xticklabels([-20,-19,-18,-17,-16,-15,-14,-13,-12,-11,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
#ax.plot([0, 40], [LFC_pvals.min(), LFC_pvals.min()], "k--")
#ax.plot([0, 40], [LFC_pvals.max(), LFC_pvals.max()], "k--")
ax.set_title("Coefficients of the LFC Ridge Regression Model")
ax.set_xlabel("Positions From TA Site")
ax.set_ylabel("Coefficients")
ax.legend()
ax.grid(True)

#######################################################################################################
#Insert Count plot
"""
#Plot the last train-test split
fig, (ax1) = plt.subplots(1, sharex=True, sharey=True)
ax1.set_title("Predicted vs. Actual log Insertion Count")
ax1.scatter(Count_y_test,Count_y_pred,s=1,c='green',alpha=0.5)
ax1.set_xlabel('Actual')
ax1.set_ylabel('Predicted')
#ax1.text(-5, 5, "R2: "+ str(sum(Count_R2_list) / len(Count_R2_list)), fontsize=10)
ax1.axhline(y=0, color='k')
ax1.axvline(x=0, color='k')
ax1.plot([-6,6], [-6,6], 'k-', alpha=0.75, zorder=1)
ax1.set_xlim(-6,6)
ax1.set_ylim(-6,6)
ax1.grid(zorder=0)
#plt.show()
"""

#Coefficients of the regression
C=[]
T=[]
G=[]
A=[]

for idx,col in enumerate(X.columns):
        if "C" in col: C.append(Count_results.coef_[idx])
        if "T" in col: T.append(Count_results.coef_[idx])
        if "G" in col: G.append(Count_results.coef_[idx])
        if "A" in col: A.append(Count_results.coef_[idx])

fig, ax = plt.subplots(figsize=(20,5))
x = np.arange(40)
bar_width = 0.2
b1 = ax.bar(x-bar_width/2 - bar_width, C,width=bar_width,label="C")
b2 = ax.bar(x-bar_width/2 , T, width=bar_width,label="T")
b3 = ax.bar(x + bar_width/2 , G, width=bar_width,label="G")
b4 = ax.bar(x + bar_width/2 + bar_width , A, width=bar_width,label="A")
ax.set_xticks(range(40))
ax.set_xticklabels([-20,-19,-18,-17,-16,-15,-14,-13,-12,-11,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
#ax.plot([0, 40], [Count_insig["IC Model Coef"].min(), Count_insig["IC Model Coef"].min()], "k--")
#ax.plot([0, 40], [Count_insig["IC Model Coef"].max(), Count_insig["IC Model Coef"].max()], "k--")
ax.set_title("Coefficients of the log Insertion Count Ridge Regression Model")
ax.set_xlabel("Positions From TA Site")
ax.set_ylabel("Coefficients")
ax.legend()
ax.grid(True)

plt.show()

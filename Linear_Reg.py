# Run Linear Regression with log Insertion Counts and LFCs

import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import sys
from sklearn.model_selection import KFold

#python3 LFC_Linear_Reg.py LFC.txt

################################################################
#read in LFC dataframe
LFC_data = pd.read_csv(sys.argv[1],sep="\t",names=["Coord","ORF ID","ORF Name","Nucl Window","State","Count","Local Mean","LFC","Description"])
sample_name = sys.argv[1].replace('_LFCs.txt','')
sample_name = sample_name.split('/')[-1]
LFC_data = LFC_data[LFC_data["State"]!="ES"] #filter out ES
LFC_data.reset_index(inplace=True, drop=True)

################################################################
#Modify Loaded Data
expanded_nucl_data = LFC_data["Nucl Window"].apply(lambda x: pd.Series(list(x)))
expanded_nucl_data.columns = [-20,-19,-18,-17,-16,-15,-14,-13,-12,-11,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,'T','A',1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]

#one-hot-encoded
expanded_nucl_data = pd.get_dummies(data=expanded_nucl_data, columns=expanded_nucl_data.columns)
expanded_nucl_data.reset_index(inplace = True, drop = True)
################################################################
#################  Regression Model  ###########################
################################################################
LFC_y = LFC_data['LFC']
Count_y = np.log10(LFC_data['Count']+0.5)
X= expanded_nucl_data.drop(["T_T","A_A"],axis=1) # one hot encoded nucl at every position except T A

#perform cross validation and train-test the models
LFC_R2_list= []
Count_R2_list=[]
kf = KFold(n_splits=10)
for train_index, test_index in kf.split(X):
	X_train, X_test = X.loc[train_index], X.loc[test_index]
	LFC_y_train, LFC_y_test = LFC_y.loc[train_index], LFC_y.loc[test_index]
	Count_y_train, Count_y_test = Count_y.loc[train_index], Count_y.loc[test_index]
	X_train = X_train.reset_index(drop=True)
	LFC_y_train = LFC_y_train.reset_index(drop=True)
	Count_y_train = Count_y_train.reset_index(drop=True)
	X_train = X_train.append(pd.DataFrame([[1 for i in range(160)]],columns=X.columns,index=[len(X_train)]))
	LFC_y_train = LFC_y_train.append(pd.Series([1]),ignore_index=True)
	Count_y_train = Count_y_train.append(pd.Series([1]),ignore_index=True)
	
	X_train = sm.add_constant(X_train)
	X_test = sm.add_constant(X_test)
	LFC_model = sm.OLS(LFC_y_train,X_train)
	LFC_results = LFC_model.fit()
	LFC_y_pred = LFC_results.predict(X_test)
	LFC_R2_list.append(r2_score(LFC_y_test, LFC_y_pred))
	Count_model = sm.OLS(Count_y_train,X_train)
	Count_results = Count_model.fit()
	Count_y_pred = Count_results.predict(X_test)
	Count_R2_list.append(r2_score(Count_y_test, Count_y_pred))


#######################################################################################################
# LFC Plot
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

#Coefficients of the regression
C=[]
T=[]
G=[]
A=[]

for idx,col in enumerate(X.columns):
        if "C" in col: C.append(LFC_results.params[idx+1])
        if "T" in col: T.append(LFC_results.params[idx+1])
        if "G" in col: G.append(LFC_results.params[idx+1])
        if "A" in col: A.append(LFC_results.params[idx+1])

fig, ax = plt.subplots(figsize=(20,5))
x = np.arange(40)
bar_width = 0.2
b1 = ax.bar(x-bar_width/2 - bar_width, C,width=bar_width,label="C")
b2 = ax.bar(x-bar_width/2 , T, width=bar_width,label="T")
b3 = ax.bar(x + bar_width/2 , G, width=bar_width,label="G")
b4 = ax.bar(x + bar_width/2 + bar_width , A, width=bar_width,label="A")
ax.set_xticks(range(40))
ax.set_xticklabels([-20,-19,-18,-17,-16,-15,-14,-13,-12,-11,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
ax.set_title("Coefficients of the LFC Linear Regression Model")
ax.set_xlabel("Positions From TA Site")
ax.set_ylabel("Coefficients")
ax.legend()
ax.grid(True)

#######################################################################################################
#Insert COunt plot
#Plot the last train-test split
fig, (ax1) = plt.subplots(1, sharex=True, sharey=True)
ax1.set_title("Predicted vs. Actual log Insertion Count")
ax1.scatter(Count_y_test,Count_y_pred,s=1,c='green',alpha=0.5)
ax1.set_xlabel('Actual')
ax1.set_ylabel('Predicted')
ax1.text(-5, 5, "R2: "+ str(sum(Count_R2_list) / len(Count_R2_list)), fontsize=10)
ax1.axhline(y=0, color='k')
ax1.axvline(x=0, color='k')
ax1.plot([-6,6], [-6,6], 'k-', alpha=0.75, zorder=1)
ax1.set_xlim(-6,6)
ax1.set_ylim(-6,6)
ax1.grid(zorder=0)
#plt.show()

#Coefficients of the regression
C=[]
T=[]
G=[]
A=[]

for idx,col in enumerate(X.columns):
        if "C" in col: C.append(Count_results.params[idx+1])
        if "T" in col: T.append(Count_results.params[idx+1])
        if "G" in col: G.append(Count_results.params[idx+1])
        if "A" in col: A.append(Count_results.params[idx+1])

fig, ax = plt.subplots(figsize=(20,5))
x = np.arange(40)
bar_width = 0.2
b1 = ax.bar(x-bar_width/2 - bar_width, C,width=bar_width,label="C")
b2 = ax.bar(x-bar_width/2 , T, width=bar_width,label="T")
b3 = ax.bar(x + bar_width/2 , G, width=bar_width,label="G")
b4 = ax.bar(x + bar_width/2 + bar_width , A, width=bar_width,label="A")
ax.set_xticks(range(40))
ax.set_xticklabels([-20,-19,-18,-17,-16,-15,-14,-13,-12,-11,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
ax.set_title("Coefficients of the log Insertion Count Linear Regression Model")
ax.set_xlabel("Positions From TA Site")
ax.set_ylabel("Coefficients")
ax.legend()
ax.grid(True)

plt.show()

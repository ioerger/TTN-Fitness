import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge
import sys
from sklearn.model_selection import KFold

"""
python3 Linear_Reg.py one_hot_encode.csv> model_descr.csv

1. Preprocess the Count/LFC data by excluding the essential sites, expanding the nucleotides surrounding the TA site, and one-hot-encode the nucleotides 
2. For each of the 10-fold X,Y split 
	i. get the X,Y train and test samples
	ii. fit the regression model to one hot encoded nucleotides with the log10 insertion counts
	iii. fit the regression model to one hot encoded nucleotides with LFCs
3. Get the Insertion Count model coefficients and pvalues and LFC model coefficents and pvalues.
4. Plot the predicted-y and test-y values of the Insertion count model and LFC model.
5. Plot the coefficents of the Insertion Count model and the LFC model with a dashed line representing the threshold of sig/insig coef. 
"""
################################################################
#read in LFC dataframe
one_hot_data = pd.read_csv(sys.argv[1])
sample_name = sys.argv[1].replace('.txt','')
sample_name = sample_name.split('/')[-1]

################################################################
#################  Regression Model  ###########################
################################################################
LFC_y = one_hot_data['LFC']
Count_y = np.log10(one_hot_data['Count']+0.5)
X= one_hot_data.drop(["T_T","A_A","Coord","Count","Local Mean","LFC","ORF ID", "ORF Name"],axis=1) # one hot encoded nucl at every position except T A
X = X.iloc[:,64:96]

print(X)
#perform cross validation and train-test the models
LFC_R2_list= []
Count_R2_list=[]
kf = KFold(n_splits=10)
for train_index, test_index in kf.split(X):
	X_train, X_test = X.loc[train_index], X.loc[test_index]
	LFC_y_train, LFC_y_test = LFC_y.loc[train_index], LFC_y.loc[test_index]
	Count_y_train, Count_y_test = Count_y.loc[train_index], Count_y.loc[test_index]
	
	#reset index of the dataframes pulled
	X_train = X_train.reset_index(drop=True)
	LFC_y_train = LFC_y_train.reset_index(drop=True)
	Count_y_train = Count_y_train.reset_index(drop=True)
	
	#adding a row of ones
	X_train = X_train.append(pd.DataFrame([[1 for i in range(len(X.columns))]],columns=X.columns,index=[len(X_train)]))
	LFC_y_train = LFC_y_train.append(pd.Series([1]),ignore_index=True)
	Count_y_train = Count_y_train.append(pd.Series([1]),ignore_index=True)
	
	# LFC Model
	
	LFC_model = Ridge(alpha=0.1)
	LFC_results = LFC_model.fit(X_train,LFC_y_train)
	LFC_y_pred = LFC_results.predict(X_test)
	LFC_R2_list.append(r2_score(LFC_y_test, LFC_y_pred)) #check getting same answers

	#Insertion Count Model
	Count_model = Ridge(alpha=0.1)
	Count_results = Count_model.fit(X_train,Count_y_train)
	Count_y_pred = Count_results.predict(X_test)
	Count_R2_list.append(r2_score(Count_y_test, Count_y_pred))

LFC_model = Ridge(alpha=0.1)
LFC_results = LFC_model.fit(X,LFC_y)
Count_model = Ridge(alpha=0.1)
Count_results = Count_model.fit(X,Count_y)

print("LFC R2: "+str(sum(LFC_R2_list) / len(LFC_R2_list)))
print("Count R2: "+ str(sum(Count_R2_list) / len(Count_R2_list)))

#######################################################################################################
# LFC Plot
#Plot the last train-test split 
fig, (ax1) = plt.subplots(1, sharex=True, sharey=True)
ax1.set_title("Predicted vs. Actual LFC")
ax1.scatter(LFC_y_test,LFC_y_pred,s=1,c='green',alpha=0.5)
ax1.set_xlabel('Actual')
ax1.set_ylabel('Predicted')
#ax1.text(-5, 5, "R2: "+ str(sum(LFC_R2_list) / len(LFC_R2_list)), fontsize=10)
ax1.axhline(y=0, color='k')
ax1.axvline(x=0, color='k')
ax1.plot([-6,6], [-6,6], 'k-', alpha=0.75, zorder=1)
ax1.grid(zorder=0)
#plt.show()

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
#x = np.arange(40)
x = np.arange(8)
bar_width = 0.2
b1 = ax.bar(x-bar_width/2 - bar_width, C,width=bar_width,label="C")
b2 = ax.bar(x-bar_width/2 , T, width=bar_width,label="T")
b3 = ax.bar(x + bar_width/2 , G, width=bar_width,label="G")
b4 = ax.bar(x + bar_width/2 + bar_width , A, width=bar_width,label="A")
#ax.set_xticks(range(40))
#ax.set_xticklabels([-20,-19,-18,-17,-16,-15,-14,-13,-12,-11,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
ax.set_xticks(range(8))
ax.set_xticklabels([-4,-3,-2,-1,1,2,3,4])
ax.set_title("Coefficients of the LFC Linear Regression Model")
ax.set_xlabel("Positions From TA Site")
ax.set_ylabel("Coefficients")
#ax.set_ylim(-0.6,0.6)
ax.legend()
ax.grid(True)

#######################################################################################################
#Insert Count plot
#Plot the last train-test split
fig, (ax1) = plt.subplots(1, sharex=True, sharey=True)
ax1.set_title("Predicted vs. Actual log Insertion Count")
ax1.scatter(Count_y_test,Count_y_pred,s=1,c='green',alpha=0.5)
ax1.set_xlabel('Actual')
ax1.set_ylabel('Predicted')
ax1.axhline(y=0, color='k')
ax1.axvline(x=0, color='k')
ax1.plot([-2,4], [-2,4], 'k-', alpha=0.75, zorder=1)
ax1.grid(zorder=0)
#plt.show()

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
#x = np.arange(40)
x = np.arange(8)
bar_width = 0.2
b1 = ax.bar(x-bar_width/2 - bar_width, C,width=bar_width,label="C")
b2 = ax.bar(x-bar_width/2 , T, width=bar_width,label="T")
b3 = ax.bar(x + bar_width/2 , G, width=bar_width,label="G")
b4 = ax.bar(x + bar_width/2 + bar_width , A, width=bar_width,label="A")
#ax.set_xticks(range(40))
#ax.set_xticklabels([-20,-19,-18,-17,-16,-15,-14,-13,-12,-11,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
ax.set_xticks(range(8))
ax.set_xticklabels([-4,-3,-2,-1,1,2,3,4])
ax.set_title("Coefficients of the log Insertion Count Linear Regression Model")
ax.set_xlabel("Positions From TA Site")
ax.set_ylabel("Coefficients")
ax.set_ylim(-0.3,0.3)
ax.legend()
ax.grid(True)

plt.show()

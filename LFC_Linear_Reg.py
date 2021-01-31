import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import sys
from sklearn.model_selection import KFold


#read in insertion counts dataframe
#expanded_data = pd.read_csv(sys.argv[1],sep=",",header=0)
#read in LFC dataframe
LFC_data = pd.read_csv(sys.argv[1],sep="\t",header=None)
LFC_data.columns= ["Coord","Nucl Window","State","Count","Local Mean","LFC"]
sample_name = sys.argv[1].replace('_LFCs.txt','')
sample_name = sample_name.split('/')[-1]

#filter out ES
LFC_data = LFC_data[LFC_data["State"]!="ES"]

expanded_data = LFC_data["Nucl Window"].apply(lambda x: pd.Series(list(x)))
expanded_data.columns = [-20,-19,-18,-17,-16,-15,-14,-13,-12,-11,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,'T','A',1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
expanded_data = expanded_data.drop([-20,-19,-18,-17,-16,-15,-14,-13,-12,-11,-10,-9,-8,-7,-6,-5,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],axis=1)
#one-hot-encoded
expanded_data = pd.get_dummies(data=expanded_data, columns=expanded_data.columns)
expanded_data["Coord"]=LFC_data["Coord"]
expanded_data["Count"]=LFC_data["Count"]
expanded_data["Local Mean"]=LFC_data["Local Mean"]
expanded_data["LFC"] = LFC_data["LFC"]

pd.set_option('display.max_colwidth',-1)
expanded_data.reset_index(inplace = True, drop = True)
np.set_printoptions(threshold=np.inf)
y = expanded_data['LFC']
X= expanded_data.drop(['Count','LFC', 'Local Mean','Coord','T_T','A_A'], axis=1)
#X= X[X.columns.drop(list(X.filter(regex='_A')))]
X =pd.DataFrame(X)
print(X)
#perform cross validation and train-test the models
R2_list= []
kf = KFold(n_splits=10)
for train_index, test_index in kf.split(X):
	X_train, X_test = X.loc[train_index], X.loc[test_index]
	y_train, y_test = y.loc[train_index], y.loc[test_index]
	X_train = X_train.reset_index(drop=True)
	y_train = y_train.reset_index(drop=True)
	X_train = X_train.append(pd.DataFrame([[1 for i in range(32)]],columns=X.columns,index=[len(X_train)]))
	y_train = y_train.append(pd.Series([1]),ignore_index=True)
	#print(X_train)
	#print(y_train)
	X_train = sm.add_constant(X_train)
	X_test = sm.add_constant(X_test)
	model = sm.OLS(y_train,X_train)#, method="qr")
	results = model.fit()
	print(results.summary())
	#coefficients.append(results.params[1:])
	y_pred = results.predict(X_test)
	R2_list.append(r2_score(y_test, y_pred))
print(R2_list)
#print(results.summary())
fig, (ax1) = plt.subplots(1, sharex=True, sharey=True)
ax1.set_title("Predicted vs. Actual LFC")
ax1.scatter(y_test,y_pred,s=1,c='green',alpha=0.5)
ax1.set_xlabel('Actual')
ax1.set_ylabel('Predicted')
ax1.text(-5, 5, "R2: "+ str(sum(R2_list) / len(R2_list)), fontsize=10)
ax1.axhline(y=0, color='k')
ax1.axvline(x=0, color='k')
ax1.plot([-6,6], [-6,6], 'k-', alpha=0.75, zorder=1)
ax1.set_xlim(-6,6)
ax1.set_ylim(-6,6)
ax1.grid(zorder=0)
#plt.show()

C=[]
T=[]
G=[]
A=[]

for idx,col in enumerate(X.columns):
        if "C" in col: C.append(results.params[idx+1])
        if "T" in col: T.append(results.params[idx+1])
        if "G" in col: G.append(results.params[idx+1])
        if "A" in col: A.append(results.params[idx+1])

fig, ax = plt.subplots(figsize=(20,5))
x = np.arange(8)
bar_width = 0.2
b1 = ax.bar(x-bar_width/2 - bar_width, C,width=bar_width,label="C")
b2 = ax.bar(x-bar_width/2 , T, width=bar_width,label="T")
b3 = ax.bar(x + bar_width/2 , G, width=bar_width,label="G")
b4 = ax.bar(x + bar_width/2 + bar_width , A, width=bar_width,label="A")
ax.set_xticks(range(8))
#ax.set_xticklabels([-20,-19,-18,-17,-16,-15,-14,-13,-12,-11,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
ax.set_xticklabels([-4,-3,-2,-1,1,2,3,4])
ax.set_title("Coefficients of the Linear Regression Model")
ax.set_xlabel("Positions From TA Site")
ax.set_ylabel("Coefficients")
ax.legend()
ax.grid(True)
plt.show()


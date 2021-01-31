import sys
import numpy as np
from sklearn.model_selection import KFold
import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from sklearn.metrics import r2_score
from scipy import stats
#create heatmap of Insertion Count correlation

#python3 Insert_Count_Linear_Reg.py <LFC Encoded file>

########################################################
###############   Correlation   ########################
########################################################
combined_wig = pd.read_csv("../data/14_replicates_combined_wig.txt",header=None,sep='\t',skiprows=17
).iloc[:,1:15]
combined_wig= pd.DataFrame(np.ma.log(combined_wig.values).filled(0), index=combined_wig.index, columns=combined_wig.columns)
corr_wigs = combined_wig.corr(method ='pearson')
# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr_wigs, dtype=bool),k=1)

# insertion counts corr across datasets figure
f, ax = plt.subplots()
f.suptitle("Pearson Correlation of log Insertion Counts across Datasets")
sns.heatmap(corr_wigs, mask=mask, cmap='Reds', vmin=0,vmax=1,square=True, linewidths=.5, cbar_kws={"shrink": .5})
#plt.show()

#t-tests output of the Correlation
f1 = []
f2=[]
corr=[]
corr_pval= []
t_val = []
ttest_pval = []
file_combos = itertools.combinations(combined_wig.columns,r=2)
for c in file_combos:
	f1.append(c[0])
	f2.append(c[1])
	corr_res = stats.pearsonr(combined_wig[c[0]],combined_wig[c[1]])
	corr.append(corr_res[0])
	corr_pval.append(corr_res[1])
	res= stats.ttest_ind(combined_wig[c[0]],combined_wig[c[1]], equal_var = False) 
	t_val.append(res[0]) 
	ttest_pval.append(res[1])
ttest_df = pd.DataFrame(data={"Wig File 1":f1,"Wig File 2":f2,"Pearson Corr":corr,"Corr Pvalue":corr_pval,"T-stat":t_val,"T-test Pval":ttest_pval})
ttest_df.to_csv("./output/Wig_Correlation_Summary.csv")

###############################################################
##############   Linear Regression  ###########################
###############################################################
#read in LFC dataframe
LFC_data = pd.read_csv(sys.argv[1],sep="\t",header=None)
LFC_data.columns= ["Coord","Nucl Window","State","Count","Local Mean","LFC"]
sample_name = sys.argv[1].replace('_LFCs.txt','')
sample_name = sample_name.split('/')[-1]

#filter out ES
LFC_data = LFC_data[LFC_data["State"]!="ES"]

expanded_data = LFC_data["Nucl Window"].apply(lambda x: pd.Series(list(x)))
expanded_data.columns = [-20,-19,-18,-17,-16,-15,-14,-13,-12,-11,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,'T','A',1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]

#one-hot-encoded
expanded_data = pd.get_dummies(data=expanded_data, columns=expanded_data.columns)
expanded_data["Coord"]=LFC_data["Coord"]
expanded_data["Count"]=LFC_data["Count"]
expanded_data["Local Mean"]=LFC_data["Local Mean"]
expanded_data["LFC"] = LFC_data["LFC"]


pd.set_option('display.max_colwidth',-1)
print(list(expanded_data.columns))
expanded_data.reset_index(inplace = True, drop = True)
np.set_printoptions(threshold=np.inf)
raw_y = expanded_data['Count'].values
y = pd.Series(np.where(raw_y != 0, np.log10(raw_y), 0)) #log counts
X= expanded_data.drop(['Count','LFC', 'Local Mean','Coord','T_T','A_A'], axis=1)
#X= X[X.columns.drop(list(X.filter(regex='_A')))]
X =pd.DataFrame(X)


#perform cross validation and train-test the models
R2_list= []
#coefficients = []
kf = KFold(n_splits=10)
for train_index, test_index in kf.split(X):
	X_train, X_test = X.loc[train_index], X.loc[test_index]
	y_train, y_test = y.loc[train_index], y.loc[test_index]
	X_train = X_train.reset_index(drop=True)
	y_train = y_train.reset_index(drop=True)
	X_train = X_train.append(pd.DataFrame([[1 for i in range(160)]],columns=X.columns,index=[len(X_train)]))
	y_train = y_train.append(pd.Series([1]),ignore_index=True)
	#print(X_train)
	#print(y_train)
	X_train = sm.add_constant(X_train)
	X_test = sm.add_constant(X_test)
	model = sm.OLS(y_train,X_train)#, method="qr")
	results = model.fit()
	#print(results.summary())
	#coefficients.append(results.params[1:])
	y_pred = results.predict(X_test)
	R2_list.append(r2_score(y_test, y_pred))
print(R2_list)
fig, (ax1) = plt.subplots(1, sharex=True, sharey=True)
ax1.set_title("Predicted vs. Actual log Insertion Counts")
ax1.scatter(y_test,y_pred,s=1,c='green',alpha=0.5)
ax1.set_xlabel('Actual')
ax1.set_ylabel('Predicted')
ax1.text(-1.5, 3.0, "R2: "+ str(round(sum(R2_list) / len(R2_list),3)), fontsize=11)
ax1.axhline(y=0, color='k')
ax1.axvline(x=0, color='k')
ax1.plot([-2,4], [-2,4], 'k-', alpha=0.75, zorder=1)
ax1.set_xlim(-2,4)
ax1.set_ylim(-2,4)
ax1.grid(zorder=0)

#coef_df = pd.DataFrame(coefficients, columns = X.columns)  
X_const = sm.add_constant(X)
model = sm.OLS(y,X_const)
results = model.fit()
print(results.summary())
print("P-values bigger than 0.05", results.pvalues[results.pvalues > 0.05])

from statsmodels.stats.multitest import fdrcorrection
p_val = fdrcorrection(results.pvalues, alpha=0.05, method='indep', is_sorted=False)[1]
print("P-values bigger than FDR", p_val[p_val > 0.05/160])
print("Their indices are ", results.pvalues.index[np.nonzero(p_val > 0.05/160)])

print("Their Coef are: ", results.params.values[np.nonzero(p_val > 0.05/160)])
print(results.params.values[np.nonzero(p_val > 0.05/160)].min(),results.params.values[np.nonzero(p_val > 0.05/160)].max())
#Model Coefficients
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
x = np.arange(40)
bar_width = 0.2
b1 = ax.bar(x-bar_width/2 - bar_width, C,width=bar_width,label="C")
b2 = ax.bar(x-bar_width/2 , T, width=bar_width,label="T")
b3 = ax.bar(x + bar_width/2 , G, width=bar_width,label="G")
b4 = ax.bar(x + bar_width/2 + bar_width , A, width=bar_width,label="A")
ax.set_xticks(range(40))
ax.set_xticklabels([-20,-19,-18,-17,-16,-15,-14,-13,-12,-11,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
ax.set_title("Coefficients of the Linear Regression Model")
ax.set_xlabel("Positions From TA Site")
ax.set_ylabel("Coefficients")
ax.legend()
ax.grid(True)
plt.show()


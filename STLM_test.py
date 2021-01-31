import sys
import pandas as pd
import statsmodels.api as sm
from statsmodels.iolib.smpickle import load_pickle
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
import math
import itertools

"""
Input: STLM pickle model, trainFile in tetraNucl file, testFile in tetraNucl form
Output: Accuracy of test data on trained STLM model
"""
import tarfile
import os

with tarfile.open(sys.argv[1]+'.tar.gz', 'r') as t:
    t.extractall('')
reg = sm.load(os.path.basename(sys.argv[1]))
os.remove(os.path.basename(sys.argv[1]))

#print(reg.params)
train_data = pd.read_csv(sys.argv[2])
train_data = train_data.dropna()
train_sample_name = sys.argv[2].replace('_tetranucl.csv','')


test_data = pd.read_csv(sys.argv[3])
test_data = test_data.dropna()
test_sample_name = sys.argv[3].replace('_tetranucl.csv','')
test_sample_name = test_sample_name.split('/')[-1]

y = test_data["LFC"]
X = test_data.drop(["Coord","Count","Local Mean","LFC"],axis=1)
X = sm.add_constant(X)
ypred = reg.predict(X)
r2score= r2_score(y,ypred)
test_data["Pred LFC"] = ypred
#print(X,y,ypred)

fig, (ax1) = plt.subplots(1, sharex=True, sharey=True)
fig.suptitle("Train: "+str(train_sample_name)+" Test: "+str(test_sample_name))
ax1.set_title("Predicted vs. Actual LFC")
ax1.scatter(y,ypred,s=1,c='green',alpha=0.5)
ax1.set_xlabel('Actual')
ax1.set_ylabel('Predicted')
ax1.text(-7, 7, "R2: "+ str(r2score), fontsize=11)
ax1.axhline(y=0, color='k')
ax1.axvline(x=0, color='k')
ax1.plot([-8,8], [-8,8], 'k--', alpha=0.75, zorder=1)
ax1.set_xlim(-8,8)
ax1.set_ylim(-8,8)
ax1.grid(zorder=0)
plt.show()

############################################################
# Correction by Correlation Coefficent
############################################################

combos=[''.join(p) for p in itertools.product(['A','C','T','G'], repeat=4)]
train_c_averages = []
test_c_averages = []
for c in combos:
	c_tetra_train = train_data[train_data[c]==1]
	train_c_averages.append(c_tetra_train["LFC"].mean())
	c_tetra_test= test_data[test_data[c]==1]
	test_c_averages.append(c_tetra_test["LFC"].mean())


cor_reg = LinearRegression(fit_intercept=True).fit(X=np.asarray(train_c_averages).reshape(-1,1),y=test_c_averages)
correction_int = cor_reg.intercept_
correction_coef = cor_reg.coef_



#print(correction_coef,correction_int)
corrected_ypred = [(i*correction_coef[0])+correction_int for i in ypred]
#corrected_ypred= [(i-correction_int)/correction_coef[0] for i in ypred]
r2score= r2_score(y,corrected_ypred)
test_data["Corrected Pred LFC"] = corrected_ypred
corrected_actual_LFC = [(i*correction_coef[0])+correction_int for i in test_data["LFC"]]
#corrected_actual_LFC = [(i-correction_int)/correction_coef[0] for i in test_data["LFC"]]
test_data["Corrected LFC"] = corrected_actual_LFC

corrected_test_c_averages = []
for c in combos:
	c_tetra_test= test_data[test_data[c]==1]
	corrected_test_c_averages.append(c_tetra_test["Corrected LFC"].mean())

corrected_train  = [(i*correction_coef[0])+correction_int for i in train_data["LFC"]]
train_data["Corrected Train LFC"] = corrected_train

corrected_train_c_averages=[]
for c in combos:
        c_tetra_train= train_data[train_data[c]==1]
        corrected_train_c_averages.append(c_tetra_train["Corrected Train LFC"].mean())

#mean_corrected_LFC = [(k/correction_coef)-correction_int for k in test_c_averages]

fig, (ax1) = plt.subplots(1, sharex=True, sharey=True)
fig.suptitle(str(train_sample_name)+"--"+str(test_sample_name)+ " TetraNucl MeanCount")
ax1.scatter(train_c_averages,test_c_averages,s=5,c='green',alpha=0.75,label="original")
ax1.scatter(corrected_train_c_averages,test_c_averages,s=5,c='blue',alpha=0.75, label='adjusted')
ax1.set_xlabel(str(train_sample_name)+' LFC Average')
ax1.set_ylabel(str(test_sample_name)+' LFC Average')
ax1.text(-3, 1.5, "Original R2: "+ str(round(r2_score(y,ypred),4)), fontsize=10)
ax1.text(-3, 1.0, "Adjusted R2: "+ str(round(r2_score(y,corrected_ypred),4)), fontsize=10)
ax1.axhline(y=0, color='k')
ax1.axvline(x=0, color='k')
ax1.plot([-3,3], [-3,3], 'k--', alpha=0.25, zorder=1)
ax1.legend()
ax1.grid(zorder=0)
#plt.show()

####################### predicted vs observed ttn #######################
pred_c_averages = []
for c in combos:
        c_tetra_test= test_data[test_data[c]==1]
        pred_c_averages.append(c_tetra_test["Pred LFC"].mean())

corrected_pred_c_averages = []
for c in combos:
        c_tetra_test= test_data[test_data[c]==1]
        corrected_pred_c_averages.append(c_tetra_test["Corrected Pred LFC"].mean())

fig, (ax1) = plt.subplots(1, sharex=True, sharey=True)
fig.suptitle("Predicted  VS. Observed "+str(test_sample_name)+ " TetraNucl MeanCount")
ax1.scatter(pred_c_averages,test_c_averages,s=5,c='green',alpha=0.75,label="original")
ax1.scatter(corrected_pred_c_averages,test_c_averages,s=5,c='blue',alpha=0.75, label='adjusted')
ax1.set_xlabel(str(test_sample_name)+' Predicted LFC Average')
ax1.set_ylabel(str(test_sample_name)+' Observed LFC Average')
ax1.axhline(y=0, color='k')
ax1.axvline(x=0, color='k')
ax1.plot([-3,3], [-3,3], 'k--', alpha=0.25, zorder=1)
ax1.legend()
ax1.text(-3, 1.5, "Original R2: "+ str(round(r2_score(y,ypred),4)), fontsize=10)
ax1.text(-3, 1.0, "Adjusted R2: "+ str(round(r2_score(y,corrected_ypred),4)), fontsize=10)
ax1.grid(zorder=0)
#plt.show()


fig, (ax2) = plt.subplots(1, sharex=True, sharey=True)
fig.suptitle("Train: "+str(train_sample_name)+" Test: "+str(test_sample_name))
ax2.set_title("CORRECTED Predicted vs. Actual LFC")
ax2.scatter(y,corrected_ypred,s=1,c='green',alpha=0.5)
ax2.set_xlabel('Actual')
ax2.set_ylabel('Corrected Predicted')
ax2.text(-7, 7, "R2: "+ str(r2score), fontsize=11)
ax2.axhline(y=0, color='k')
ax2.axvline(x=0, color='k')
ax2.plot([-8,8], [-8,8], 'k--', alpha=0.75, zorder=1)
ax2.set_xlim(-8,8)
ax2.set_ylim(-8,8)
ax2.grid(zorder=0)
#plt.show()


############################################################
# Predicted vs Observed Counts Graph
############################################################
def calcPredictedCounts(row):
	predCount = row["Local Mean"]*math.pow(2,row["Pred LFC"])
	return predCount
def calcAdjustedPredictedCounts(row):
	predCount = row["Local Mean"]*math.pow(2,row["Corrected Pred LFC"])
	return predCount
test_data["Predicted Count"]=test_data.apply(calcPredictedCounts,axis=1)
test_data["Adjusted Predicted Count"]=test_data.apply(calcAdjustedPredictedCounts,axis=1)

fig, (ax3) = plt.subplots(1, sharex=True, sharey=True)
fig.suptitle(str(test_sample_name)+" Observed vs Predicted Counts")
ax3.scatter(np.log10(test_data["Count"]),np.log10(test_data["Predicted Count"]),s=1,c='green',alpha=0.5,label='original')
ax3.scatter(np.log10(test_data["Count"]),np.log10(test_data["Adjusted Predicted Count"]),s=1,c='blue',alpha=0.5,label='corrected')
ax3.set_xlabel('log Observed Count')
ax3.set_ylabel('log Predicted Count')
ax3.axhline(y=0, color='k')
ax3.axvline(x=0, color='k')
#ax1.plot([-8,8], [-8,8], 'k--', alpha=0.75, zorder=1)
#ax1.set_xlim(-8,8)
#ax1.set_ylim(-8,8)
ax3.grid(zorder=0)
ax3.legend()
plt.show()


data = test_data.to_csv(header=True, index=False).split('\n')
vals = '\n'.join(data)
print(vals)

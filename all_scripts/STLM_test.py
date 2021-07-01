import sys,os,tarfile
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
python3 STLM_test.py model.pickle trainTTN.csv testTTN.csv > predictions.csv

1. Load in model from the pickle file
2. Load in the csv
3. Predict test data LFC using the loaded model
4. Plot overall predicted vs. actual LFC of the test data
5. Plot the mean predicted vs. mean actual LFC
6. Plot Predicted vs. Actual Counts
7. Output the compilation of data
"""
with tarfile.open(sys.argv[1]+'.tar.gz', 'r') as t:
    t.extractall('')
reg = sm.load(sys.argv[1])
os.remove(sys.argv[1])

raw_train_data = pd.read_csv(sys.argv[2])
train_data = raw_train_data[raw_train_data["State"]!="ES"]
train_data = train_data.dropna()
train_sample_name = sys.argv[2].split("/")[-1].replace('.csv','')

raw_test_data = pd.read_csv(sys.argv[3])
test_data = raw_test_data[raw_test_data["State"]!="ES"]
test_data = test_data.dropna()
test_sample_name = sys.argv[3].split('/')[-1].replace('.csv','')

y = test_data["LFC"]
X_raw = raw_test_data.drop(["Coord","Count","ORF ID","ORF Name","Local Mean","LFC","State"],axis=1)
X_raw = sm.add_constant(X_raw)
ypred_raw = reg.predict(X_raw)

X = test_data.drop(["Coord","Count","ORF ID","ORF Name","Local Mean","LFC","State"],axis=1)
X = sm.add_constant(X)
ypred = reg.predict(X)
r2score= r2_score(y,ypred)

raw_test_data["Pred LFC"] = ypred_raw
test_data["Pred LFC"] = ypred

###################### Corrections ############################ 
combos=[''.join(p) for p in itertools.product(['A','C','T','G'], repeat=4)]
train_c_averages = []
test_c_averages = []
pred_c_averages=[]

for c in combos:
        c_tetra_train = train_data[train_data[c]==1]
        train_c_averages.append(c_tetra_train["LFC"].mean())
        c_tetra_test= test_data[test_data[c]==1]
        test_c_averages.append(c_tetra_test["LFC"].mean())
        pred_c_averages.append(c_tetra_test["Pred LFC"].mean())

cor_reg = LinearRegression(fit_intercept=True).fit(X=np.asarray(train_c_averages).reshape(-1,1),y=test_c_averages)
correction_int = cor_reg.intercept_
correction_coef = cor_reg.coef_

corrected_ypred = [(i*correction_coef[0])+correction_int for i in ypred]
r2score= r2_score(y,corrected_ypred)
test_data["Corrected Pred LFC"] = corrected_ypred
raw_test_data["Corrected Pred LFC"] = [(i*correction_coef[0])+correction_int for i in ypred_raw]

corrected_actual_LFC = [(i*correction_coef[0])+correction_int for i in test_data["LFC"]]
test_data["Corrected LFC"] = corrected_actual_LFC
raw_test_data["Corrected LFC"] = [(i*correction_coef[0])+correction_int for i in raw_test_data["LFC"]]

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

############################################################
# Plot predicted vs. actual LFC
############################################################
fig, (ax1) = plt.subplots(1, sharex=True, sharey=True)
fig.suptitle("Test Genome: "+str(test_sample_name))
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
#plt.show()


############################################################
# Predicted vs Observed Counts Graph
############################################################
def calcPredictedCounts(row):
	predCount = row["Local Mean"]*math.pow(2,row["Pred LFC"])
	return predCount
test_data["Predicted Count"]=test_data.apply(calcPredictedCounts,axis=1)

fig, (ax3) = plt.subplots(1, sharex=True, sharey=True)
fig.suptitle(str(test_sample_name)+" Observed vs Predicted Counts")
ax3.scatter(np.log10(test_data["Count"]),np.log10(test_data["Predicted Count"]),s=1,c='green',alpha=0.5)
ax3.set_xlabel('log Observed Count')
ax3.set_ylabel('log Predicted Count')
ax3.axhline(y=0, color='k')
ax3.axvline(x=0, color='k')
ax3.grid(zorder=0)
ax3.legend()
#plt.show()

############################################################
# Corrected Graphs
############################################################
##### Correction Scatter Graph #######
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
####### Corrected Predicted vs Actual LFC ########
fig, (ax1) = plt.subplots(1, sharex=True, sharey=True)
fig.suptitle("Test Genome: "+str(test_sample_name))
ax1.set_title("CORRECTED Predicted vs. Actual LFC")
ax1.scatter(y,corrected_ypred,s=1,c='green',alpha=0.5)
ax1.set_xlabel('Actual')
ax1.set_ylabel('Corrected Predicted')
ax1.text(-7, 7, "R2: "+ str(r2score), fontsize=11)
ax1.axhline(y=0, color='k')
ax1.axvline(x=0, color='k')
ax1.plot([-8,8], [-8,8], 'k--', alpha=0.75, zorder=1)
ax1.set_xlim(-8,8)
ax1.set_ylim(-8,8)
ax1.grid(zorder=0)
plt.show()

# print to output
data = raw_test_data.to_csv(header=True, index=False).split('\n')
vals = '\n'.join(data)
print(vals)


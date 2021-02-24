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
python3 STLM_test.py model.pickle testTTN.csv > predictions.csv

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
reg = sm.load(os.path.basename(sys.argv[1]))
os.remove(os.path.basename(sys.argv[1]))

test_data = pd.read_csv(sys.argv[2])
test_data = test_data.dropna()
test_sample_name = sys.argv[2].replace('.csv','')
test_sample_name = test_sample_name.split('/')[-1]

y = test_data["LFC"]
X = test_data.drop(["Coord","Count","Local Mean","LFC"],axis=1)
X = sm.add_constant(X)
ypred = reg.predict(X)
r2score= r2_score(y,ypred)
test_data["Pred LFC"] = ypred

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
# Plot predicted vs. actual LFC by TTN
############################################################
combos=[''.join(p) for p in itertools.product(['A','C','T','G'], repeat=4)]
test_c_averages = []
for c in combos:
        c_tetra_test= test_data[test_data[c]==1]
        test_c_averages.append(c_tetra_test["LFC"].mean())

pred_c_averages = []
for c in combos:
        c_tetra_test= test_data[test_data[c]==1]
        pred_c_averages.append(c_tetra_test["Pred LFC"].mean())


fig, (ax1) = plt.subplots(1, sharex=True, sharey=True)
fig.suptitle("Predicted  VS. Observed "+str(test_sample_name)+ " TetraNucl MeanCount")
ax1.scatter(pred_c_averages,test_c_averages,s=5,c='green',alpha=0.75)
ax1.set_xlabel(str(test_sample_name)+' Predicted LFC Average')
ax1.set_ylabel(str(test_sample_name)+' Observed LFC Average')
ax1.axhline(y=0, color='k')
ax1.axvline(x=0, color='k')
ax1.plot([-3,3], [-3,3], 'k--', alpha=0.25, zorder=1)
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
plt.show()

# print to output
data = test_data.to_csv(header=True, index=False).split('\n')
vals = '\n'.join(data)
print(vals)

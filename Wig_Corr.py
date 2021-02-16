import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
#create heatmap of Insertion Count correlation

#python3 Insert_Count_Linear_Reg.py combinedwig text > wigcorrelationCSV

combined_wig = pd.read_csv(sys.argv[1],header=None,sep='\t',skiprows=17).iloc[:,1:15]
combined_wig= pd.DataFrame(np.ma.log(combined_wig.values).filled(0), index=combined_wig.index, columns=combined_wig.columns)
corr_wigs = combined_wig.corr(method ='pearson')
# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr_wigs, dtype=bool),k=1)

# insertion counts corr across datasets figure
f, ax = plt.subplots()
f.suptitle("Pearson Correlation of log Insertion Counts across Datasets")
sns.heatmap(corr_wigs, mask=mask, cmap='Reds', vmin=0,vmax=1,square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.show()

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
data = ttest_df.to_csv(header=True, index=False).split('\n')
vals = '\n'.join(data)
print(vals)


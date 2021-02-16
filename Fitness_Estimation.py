import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import scipy
import seaborn as sns
from scipy.signal import convolve as scipy_convolve
import os,sys,tarfile
import statsmodels.stats.multitest  
from matplotlib.lines import Line2D

#python3 Fitness_Estimation.py LFC.txt STLM_prediction.csv HMMStages.csv model.pickle prot_table

##################################################################################
# Read in Data
TA_site_data = pd.read_csv(sys.argv[1],sep="\t",names=["Coord","ORF ID","ORF Name","Nucleotide","State","Count","Local Mean","LFC","Description"])
ttn_data = pd.read_csv(sys.argv[2])
hmm_stages = pd.read_csv(sys.argv[3])
hmm_stages.set_index("ORF ID", drop=True, inplace=True)

with tarfile.open(sys.argv[4]+'.tar.gz', 'r') as t:
    t.extractall('')
reg = sm.load(os.path.basename(sys.argv[4]))
os.remove(os.path.basename(sys.argv[4]))

prot_table = pd.read_csv(sys.argv[5],sep='\t',header=None, names= ["Description", "X1","X2","X3","X4","X5","X6","ORF Name","ORF ID","X9","X10"])
prot_table.set_index("ORF ID", drop=True, inplace=True)
###################################################################################
#Filter Loaded  Data
def filter_ES(data_df):
	all_genes = data_df["ORF ID"].unique()
	genes_to_keep=[]
	for g in all_genes:
		sub_df=data_df[data_df["ORF ID"]==g]
		state_lbs = sub_df["State"].to_list()
		percentage = state_lbs.count('NE')/len(state_lbs)
		if percentage>0.5:
			genes_to_keep.append(g)
	filtered_df = data_df[data_df["ORF ID"].isin(genes_to_keep)] 
	return filtered_df

filtered_ES_gene_data = filter_ES(TA_site_data) #sites from genes that are mostly non-ES
ttn_filtered_data = ttn_data[ttn_data["Coord"].isin(filtered_ES_gene_data["Coord"])] #ttn data only from sites within nonES genes
filtered_TA_sites_data = filtered_ES_gene_data[filtered_ES_gene_data["Coord"].isin(ttn_filtered_data["Coord"])] #essentially taking all the remaining ES sites out.
ttn_filtered_data=ttn_filtered_data.reset_index(drop=True)
filtered_TA_sites_data = filtered_TA_sites_data.reset_index(drop=True)
ttn_filtered_data["ORF ID"] = filtered_TA_sites_data["ORF ID"]
##########################################################################################
#Linear Regression

gene_one_hot_encoded= pd.get_dummies(filtered_TA_sites_data["ORF ID"],prefix='')
ttn_vectors = ttn_filtered_data.drop(["Coord","Count","Local Mean","LFC","Pred LFC","Predicted Count", "ORF ID"],axis=1)

X1 = pd.concat([gene_one_hot_encoded],axis=1)
X1 = sm.add_constant(X1)
X2 = pd.concat([gene_one_hot_encoded,ttn_vectors],axis=1)
X2 = sm.add_constant(X2)
Y = np.log10(ttn_filtered_data["Count"]+0.5)

results1 = sm.OLS(Y,X1).fit()
results2 = sm.OLS(Y,X2).fit()

##########################################################################################
#create Models Summary df 
def calcpval(row):
        c1 = row["M0 Coef"]
        c2 = row["M1 Coef"]
        g = row.name.split("_")[1]
        n = len(filtered_TA_sites_data[filtered_TA_sites_data["ORF ID"]==g])
        se1 = results1.bse["_"+g]/np.sqrt(n)
        se2 = results2.bse["_"+g]/np.sqrt(n)
        vpooled = (se1**2.0 + se2**2.0)
        tt = (c1-c2)/np.sqrt(vpooled) #null hypothesis: c1==c2 -> c1-c2=0
        df = n + n - 2
        diff_pval = (1 - scipy.stats.t.cdf(abs(tt), df)) * 2  #Other side of the curve. NOT times 2 but maybe 1-CDF. Abs TT
        return diff_pval

Models_df = pd.DataFrame(results1.params[1:],columns=["M0 Coef"])
Models_df["M0 Pval"] = results1.pvalues[1:]
Models_df["M0 Adjusted Pval"] = statsmodels.stats.multitest.fdrcorrection(results1.pvalues[1:],alpha=0.05)[1]
Models_df["M1 Coef"]= results2.params[1:-256]
Models_df["M1 Pval"] = results2.pvalues[1:-256]
Models_df["M1 Adjusted Pval"] = statsmodels.stats.multitest.fdrcorrection(results2.pvalues[1:-256],alpha=0.05)[1]
Models_df["Coef Diff (M1-M0)"] = Models_df["M1 Coef"] - Models_df["M0 Coef"]
Models_df["Coef Diff Pval"] = Models_df.apply(calcpval, axis = 1)
#creating a mask for the adjusted pvals
p = Models_df["Coef Diff Pval"]
mask = np.isfinite(p)
pval_corrected = np.full(p.shape, np.nan)
pval_corrected[mask] = statsmodels.stats.multitest.fdrcorrection(p[mask], alpha=0.05)[1]
Models_df["Coef Diff Adjusted Pval"] = pval_corrected

Models_df["Gene+TTN States"] = "ES/ESD"
Models_df.loc[(Models_df["M1 Coef"]>0) & (Models_df["M1 Adjusted Pval"]<0.05),"Gene+TTN States"]="GA"
Models_df.loc[(Models_df["M1 Coef"]<0) & (Models_df["M1 Adjusted Pval"]<0.05),"Gene+TTN States"]="GD"
Models_df.loc[(Models_df["M1 Coef"]==0) & (Models_df["M1 Adjusted Pval"]<0.05),"Gene+TTN States"]="NE"
Models_df.loc[(Models_df["M1 Adjusted Pval"]>0.05),"Gene+TTN States"]="NE"


#########################################################################################
#Write Models Information to CSV
# Columns: ORF ID, ORF Name, ORF Description,M0 Coef, M0 Adj Pval
gene_dict={} #dictionary to map information per gene
for g in prot_table.index.unique():
	val = [g, prot_table.loc[g,"ORF Name"], prot_table.loc[g,"Description"],len(TA_site_data[TA_site_data["ORF ID"]==g]),len(TA_site_data[(TA_site_data["ORF ID"]==g) & (TA_site_data["Count"]==0)])]
	if "_"+g in results1.params[1:]:
		pred_counts = ttn_filtered_data[ttn_filtered_data["ORF ID"]==g]["Predicted Count"]
		actual_counts = ttn_filtered_data[ttn_filtered_data["ORF ID"]==g]["Count"]
		val.extend([True,Models_df.loc["_"+g,"M0 Coef"],Models_df.loc["_"+g,"M0 Adjusted Pval"],Models_df.loc["_"+g,"M1 Coef"],Models_df.loc["_"+g,"M1 Adjusted Pval"], Models_df.loc["_"+g,"Coef Diff (M1-M0)"], Models_df.loc["_"+g,"Coef Diff Adjusted Pval"],np.mean(pred_counts),np.mean(actual_counts), Models_df.loc["_"+g,"Gene+TTN States"]])
	
	else:
		actual_counts = filtered_TA_sites_data[filtered_TA_sites_data["ORF ID"]==g]["Count"]
		val.extend([False,None, None, None, None, None,None,None,actual_counts,"ES/ESD"])

	if g in hmm_stages.index:
		val.extend([hmm_stages.loc[g,"Final Call"]])
	else:
		val.extend(["Uncertain"])
	gene_dict[g] = val
gene_df = pd.DataFrame.from_dict(gene_dict,orient='index')
gene_df.columns=["ORF ID","Name","Description","Total # TA Sites","#Sites with 0 insertions","Used in Models","Gene (M0) Coef","Gene(M0) Adj Pval","Gene+TTN (M1) Coef","Gene+TTN (M1) Adj Pval","Coef Diff (M1-M0)","Coef Diff Adj Pval","Mean STLM Predicted Count","Mean Actual Count","Gene+TTN States","HMM+NP States"]
print(gene_df)

gene_data = gene_df.to_csv(header=True, index=False).split('\n')
vals = '\n'.join(gene_data)
print(vals)


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
from sklearn.metrics import confusion_matrix

"""
python3 Fitness_Estimation.py LFC.txt STLM_prediction.csv HMMStages.csv model.pickle prot_table Gumbel_pred.txt> Gene_estimation.csv

1. Read in Data
2. Filter out data to include the coordinates in both the TTN csv and the gene data (exclude the genes labeled ES and TA sites labeled ES)
3. Linear Regression with Gene-only and Gene+TTN data
4. Fitness estimations based on Gene+TTN model
5. Create Gene Summary dataframe output
"""
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

gumbel_pred = pd.read_csv(sys.argv[6],sep='\t')
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

def gumbel_filter_ES(data_df):
	all_genes = data_df["ORF ID"].unique()
	genes_to_keep=[]
	for g in all_genes:
		gumbel_call='E'
		sub_gumbel=gumbel_pred[gumbel_pred["Orf"]==g]
		if len(sub_gumbel)>0:gumbel_call = sub_gumbel["Call"].iloc[0]
		if gumbel_call!="E": genes_to_keep.append(g)
	filtered_df = data_df[data_df["ORF ID"].isin(genes_to_keep)] 
	return filtered_df

#filtered_ES_gene_data = filter_ES(TA_site_data) #sites from genes that are mostly non-ES
filtered_ES_gene_data = gumbel_filter_ES(TA_site_data)
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
gene_df.columns=["ORF ID","Name","Description","Total # TA Sites","#Sites with 0 insertions","Used in Models","Gene (M0) Coef","Gene (M0) Adj Pval","Gene+TTN (M1) Coef","Gene+TTN (M1) Adj Pval","Coef Diff (M1-M0)","Coef Diff Adj Pval","Mean STLM Predicted Count","Mean Actual Count","Gene+TTN States","HMM+NP States"]
#print(gene_df)

gene_data = gene_df.to_csv(header=True, index=False).split('\n')
vals = '\n'.join(gene_data)
print(vals)
##########################################################################################
# FIGURES
filtered_gene_df = gene_df[gene_df["Used in Models"]==True]

g = sns.jointplot(data=filtered_gene_df, x="Gene (M0) Coef",y="Gene+TTN (M1) Coef",hue="HMM+NP States", alpha=0.75,palette = dict({'Uncertain':'#fdc086','NE': '#386cb0','ES': '#beaed4','ESD': '#beaed4','GD': '#f0027f', 'GA':'#7fc97f'}))
g.plot_marginals(sns.histplot,bins=50,kde=True)
g.ax_joint.text(gene_df.loc["Rv3461c","Gene (M0) Coef"]+0.05, gene_df.loc["Rv3461c","Gene+TTN (M1) Coef"]-0.025,"Rv3461c",bbox={'facecolor': 'white', 'alpha': 0.75, 'pad': 2},horizontalalignment='left')
g.ax_joint.text(gene_df.loc["Rv0833","Gene (M0) Coef"]+0.05, gene_df.loc["Rv0833","Gene+TTN (M1) Coef"]-0.025,"Rv0833",bbox={'facecolor': 'white', 'alpha': 0.75, 'pad': 2},horizontalalignment='left')
g.ax_joint.set(xlim=(-3.5,2.5),ylim=(-3.5,2.5))
g.ax_joint.plot([-3.5,2.5], [-3.5,2.5], ':k',alpha=0.4)

g.ax_joint.scatter([gene_df.loc["Rv3461c","Gene (M0) Coef"],gene_df.loc["Rv0833","Gene (M0) Coef"]],[gene_df.loc["Rv3461c","Gene+TTN (M1) Coef"],gene_df.loc["Rv0833","Gene+TTN (M1) Coef"]], alpha=1.0, color="none",edgecolor="black")
g.ax_joint.legend(handles = [Line2D([0], [0], marker='o', color='w', label='NE', markerfacecolor='#386cb0', markersize=5),
Line2D([0], [0], marker='o', color='w', label='ES/ESD', markerfacecolor='#beaed4', markersize=5),
Line2D([0], [0], marker='o', color='w', label='GD', markerfacecolor='#f0027f', markersize=5),
Line2D([0], [0], marker='o', color='w', label='GA', markerfacecolor='#7fc97f', markersize=5),
Line2D([0], [0], marker='o', color='w', label='Uncertain', markerfacecolor='#fdc086', markersize=5)],
title="HMM+NP States")

#plt.show()

############## Model 2 Volcano #############
gene_df["States"] = gene_df["HMM+NP States"]
gene_df.loc[gene_df["Gene+TTN (M1) Adj Pval"] > 0.05, 'States'] = "Insig"
color_dict = dict({'Insig':'gray','Uncertain':'#fdc086',
                   'NE': '#386cb0',
                   'ES': '#beaed4','ESD': '#beaed4',
                   'GD': '#f0027f', 'GA':'#7fc97f'})
plt.figure()
g=sns.scatterplot(x=gene_df["Gene+TTN (M1) Coef"],y=0-np.log10(gene_df["Gene+TTN (M1) Adj Pval"]),hue=gene_df["States"],alpha=0.75,palette=color_dict)
g.set(xlabel="Gene+TTN Model Gene Coef", ylabel="-log10 (Gene+TTN Model Adjusted Pval)",xlim=(-2.5,1.5),ylim=(0,50))
g.set_title("Gene+TTN Model")
g.legend(handles = [Line2D([0], [0], marker='o', color='w', label='NE', markerfacecolor='#386cb0', markersize=5),
Line2D([0], [0], marker='o', color='w', label='ES/ESD', markerfacecolor='#beaed4', markersize=5),
Line2D([0], [0], marker='o', color='w', label='GD', markerfacecolor='#f0027f', markersize=5),
Line2D([0], [0], marker='o', color='w', label='GA', markerfacecolor='#7fc97f', markersize=5),
Line2D([0], [0], marker='o', color='w', label='Uncertain', markerfacecolor='#fdc086', markersize=5)],
title="HMM+NP States")
#g.plot([-2.5,0-np.log10(0.05)], [2.5, 0-np.log10(0.05)], ':k',alpha=0.4)
g.axhline(0-np.log10(0.05), ls='--',color="black",alpha=0.4)
g.axvline(0,lw=3,color="black",alpha=0.4)
g.text(gene_df.loc["Rv3461c","Gene+TTN (M1) Coef"]+0.05, 0-np.log10(gene_df.loc["Rv3461c","Gene+TTN (M1) Adj Pval"])-0.025,"Rv3461c",bbox={'facecolor': 'white', 'alpha': 0.75, 'pad': 2},horizontalalignment='left')
g.text(gene_df.loc["Rv0833","Gene+TTN (M1) Coef"]+0.05, 0-np.log10(gene_df.loc["Rv0833","Gene+TTN (M1) Adj Pval"])-0.025,"Rv0833",bbox={'facecolor': 'white', 'alpha': 0.75, 'pad': 2},horizontalalignment="left")
g.scatter([gene_df.loc["Rv3461c","Gene+TTN (M1) Coef"],gene_df.loc["Rv0833","Gene+TTN (M1) Coef"]],[0-np.log10(gene_df.loc["Rv3461c","Gene+TTN (M1) Adj Pval"]),0-np.log10(gene_df.loc["Rv0833","Gene+TTN (M1) Adj Pval"])], color="none",edgecolor="black")

#plt.show()

############################################################################################################################
gene_df.loc[gene_df["HMM+NP States"]=="ES","HMM+NP States"] = "ES/ESD"
gene_df.loc[gene_df["HMM+NP States"]=="ESD","HMM+NP States"] = "ES/ESD"

prev_states = gene_df["HMM+NP States"]
new_states = gene_df["Gene+TTN States"]

cat = ["GA","NE","GD","ES/ESD","Uncertain"] 
conf_matrix = confusion_matrix(prev_states, new_states,labels=cat)
conf_matrix = pd.DataFrame(conf_matrix, index=cat, columns=cat)
plt.figure()
sns.heatmap(conf_matrix, linewidths=1, linecolor="black",annot=True, fmt='g',vmin=3000,cmap="Greys",annot_kws={"fontsize":12}).set(ylabel="HMM+NP States",xlabel="Gene+TTN States")
plt.show()

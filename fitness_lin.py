import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import seaborn as sns
from scipy.signal import convolve as scipy_convolve
import os,sys,tarfile
import statsmodels.stats.multitest  
from matplotlib.lines import Line2D

def filter_ES(data_df):
	all_genes = data_df["gene"].unique()
	genes_to_keep=[]
	TA_dict={}
	for g in all_genes:
		sub_df=data_df[data_df["gene"]==g]
		state_lbs = sub_df["state"].to_list()
		percentage = state_lbs.count('NE')/len(state_lbs)
		if percentage>0.5:
			genes_to_keep.append(g)
		TA_dict[g]=[len(state_lbs),state_lbs.count('ES'), True if percentage>0.5 else False]
	filtered_df = data_df[data_df["gene"].isin(genes_to_keep)] 
	return filtered_df,TA_dict


#Data Set up
base_data = pd.read_csv("H37RvBD1_LFC_mod.txt",sep="\t",names=["Coord","gene","nucl","state","Count","Local Mean","LFC"])
#nonES = base_data[base_data["state"]!="ES"]
filtered_ES_gene_data,TA_dict = filter_ES(base_data)
ttn_data = pd.read_csv("STLM_h37rvbd1.csv")
hmm_states = pd.read_csv("hmm_stages.csv")
prot_table = pd.read_csv("H37RvBD1.prot_table",sep='\t',header=None)

with tarfile.open('H37RvBD1.pickle.tar.gz', 'r') as t:
    t.extractall('')
STLM_model = sm.load(os.path.basename('H37RvBD1.pickle'))
os.remove(os.path.basename('H37RvBD1.pickle'))

merged_data = ttn_data[ttn_data["Coord"].isin(filtered_ES_gene_data["Coord"])] #ttn data is only NE sites and does not include mostly ES genes.
filtered_ttn_data = filtered_ES_gene_data[filtered_ES_gene_data["Coord"].isin(merged_data["Coord"])]
merged_data=merged_data.reset_index(drop=True)
filtered_ttn_data = filtered_ttn_data.reset_index(drop=True)
merged_data["Gene"] = filtered_ttn_data["gene"]


gene_data = pd.get_dummies(merged_data["Gene"],prefix='')
STLM_data = merged_data.drop(["Coord","Count","Local Mean","LFC","Gene","Corrected LFC","Adjusted Predicted Count","Pred LFC","Corrected Pred LFC","Predicted Count"],axis=1)
Local_Avg = pd.DataFrame(merged_data["Local Mean"],dtype=float)
Local_Avg = Local_Avg.reset_index(drop=True)

X1 = pd.concat([gene_data],axis=1)
X1 = sm.add_constant(X1)
X2 = pd.concat([gene_data,STLM_data],axis=1)
X2 = sm.add_constant(X2)
Y = np.log10(merged_data["Count"]+0.5)

results1 = sm.OLS(Y,X1).fit()
results2 = sm.OLS(Y,X2).fit()

Model2_gene_pvalues = pd.DataFrame(results2.pvalues[1:-256],columns=["pvalues"])
Model2_gene_pvalues["adj_pvalues"] = statsmodels.stats.multitest.fdrcorrection(results2.pvalues[1:-256])[1]
Model1_gene_pvalues = pd.DataFrame(results1.pvalues[1:],columns=["pvalues"])
Model1_gene_pvalues["adj_pvalues"] = statsmodels.stats.multitest.fdrcorrection(results1.pvalues[1:])[1]

descr_dict={}
for idx,row in hmm_states.iterrows():
	descr_dict[row["ORF ID"]] = [row["Name"],row["Description"],row["Final Call"]]
descr_dict['igr']=['igr','igr',"Uncertain"]

prot_table_dict={}
for idx,row in prot_table.iterrows():
	prot_table_dict[row[8]]= [row[7],row[0],"Uncertain"]

count_dict={}
for g in TA_dict.keys():
	sub_base_data=base_data[base_data["gene"]==g]
	sub_ttn = ttn_data[ttn_data["Coord"].isin(sub_base_data["Coord"])]
	count_dict[g] = [sub_ttn["Predicted Count"].mean(),sub_base_data["Count"].mean()]

############## WRITE MODELS #########
gene_dict={}
for g in TA_dict.keys():
	c1=None
	c2=None
	diff=None
	pval1 =None
	pval2=None
	if "_"+g in results1.params:
		pval1=-np.log10(Model1_gene_pvalues.loc["_"+g,"adj_pvalues"])
		pval2=-np.log10(Model2_gene_pvalues.loc["_"+g,"adj_pvalues"])
		c1 = results1.params["_"+g]
		c2 = results2.params["_"+g]

	gene_dict[g]=[c1,c2,count_dict[g][0],count_dict[g][1],count_dict[g][1]-count_dict[g][0]]
	gene_dict[g].extend(TA_dict[g])
	if g not in descr_dict: 
		gene_dict[g] = [g,prot_table_dict[g][0],prot_table_dict[g][1]]+gene_dict[g]
		gene_dict[g].extend([prot_table_dict[g][2],pval1,pval2])

	else: 
		gene_dict[g] = [g,descr_dict[g][0],descr_dict[g][1]]+gene_dict[g]
		gene_dict[g].extend([descr_dict[g][2],pval1,pval2])


gene_df = pd.DataFrame.from_dict(gene_dict,orient='index')
gene_df.columns=["ORF ID","Name","Description","Gene Coef","Gene+TTN Coef","Predicted Count","Actual Count","Count Diff","Total TA Sites","#ES","Used","Prev States","Gene Adj Pval", "Gene+TTN Adj Pval"]
gene_df = gene_df.set_index ("ORF ID")

################################################
################ FIGURES #######################
################################################

######### Gene Coef Corr ###############
g = sns.jointplot(data=gene_df, x="Gene Coef",y="Gene+TTN Coef",hue="Prev States", alpha=0.75,palette = dict({'Uncertain':'#fdc086','NE': '#386cb0','ES': '#beaed4','ESD': '#beaed4','GD': '#f0027f', 'GA':'#7fc97f'}))
g.plot_marginals(sns.histplot,bins=50,kde=True)
g.ax_joint.text(gene_df.loc["Rv3461c","Gene Coef"]+0.05, gene_df.loc["Rv3461c","Gene+TTN Coef"]-0.025,"Rv3461c",bbox={'facecolor': 'white', 'alpha': 0.75, 'pad': 2},horizontalalignment='left')
g.ax_joint.text(gene_df.loc["Rv2520c","Gene Coef"]+0.05, gene_df.loc["Rv2520c","Gene+TTN Coef"]-0.025,"Rv2520c",bbox={'facecolor': 'white', 'alpha': 0.75, 'pad': 2},horizontalalignment='left')
g.ax_joint.text(gene_df.loc["Rv0833","Gene Coef"]+0.05, gene_df.loc["Rv0833","Gene+TTN Coef"]-0.025,"Rv0833",bbox={'facecolor': 'white', 'alpha': 0.75, 'pad': 2},horizontalalignment='left')

g.ax_joint.scatter([gene_df.loc["Rv3461c","Gene Coef"],gene_df.loc["Rv2520c","Gene Coef"],gene_df.loc["Rv0833","Gene Coef"]],[gene_df.loc["Rv3461c","Gene+TTN Coef"],gene_df.loc["Rv2520c","Gene+TTN Coef"],gene_df.loc["Rv0833","Gene+TTN Coef"]], alpha=1.0, color="none",edgecolor="black")

g.ax_joint.legend(handles = [Line2D([0], [0], marker='o', color='w', label='NE', markerfacecolor='#386cb0', markersize=5),
Line2D([0], [0], marker='o', color='w', label='ES/ESD', markerfacecolor='#beaed4', markersize=5),
Line2D([0], [0], marker='o', color='w', label='GD', markerfacecolor='#f0027f', markersize=5),
Line2D([0], [0], marker='o', color='w', label='GA', markerfacecolor='#7fc97f', markersize=5),
Line2D([0], [0], marker='o', color='w', label='Uncertain', markerfacecolor='#fdc086', markersize=5)],
title="HMM+NP States")

plt.show()

############## TTN Coef Corr ############
g = sns.jointplot(y=results2.params[-256:], x=STLM_model.params[1:],marginal_kws=dict(bins=50,kde=True))
#g.fig.suptitle("Correlation of TTN Coefficents Between Models")
g.ax_joint.set_xlabel("Gene+TTN TTN Coef")
g.ax_joint.set_ylabel("STLM Coef")
plt.show()

############ Model 1 Volcano #############
gene_df["States"] = gene_df["Prev States"]
gene_df.loc[gene_df["Gene Adj Pval"]< 0-np.log10(0.05), 'States'] = "Insig"
color_dict = dict({'Insig':'gray','Uncertain':'#fdc086',
                   'NE': '#386cb0',
                   'ES': '#beaed4','ESD': '#beaed4',
                   'GD': '#f0027f', 'GA':'#7fc97f'})
g=sns.scatterplot(data=gene_df, x="Gene Coef",y="Gene Adj Pval",hue="States",alpha=0.75,palette=color_dict)
g.set_title("Gene-Only Method")
g.set(xlabel="Gene-Only Gene Coef", ylabel="-log10 (Gene-Only Adjusted Pval)")
g.legend(handles = [Line2D([0], [0], marker='o', color='w', label='NE', markerfacecolor='#386cb0', markersize=5),
Line2D([0], [0], marker='o', color='w', label='ES/ESD', markerfacecolor='#beaed4', markersize=5),
Line2D([0], [0], marker='o', color='w', label='GD', markerfacecolor='#f0027f', markersize=5),
Line2D([0], [0], marker='o', color='w', label='GA', markerfacecolor='#7fc97f', markersize=5),
Line2D([0], [0], marker='o', color='w', label='Uncertain', markerfacecolor='#fdc086', markersize=5)],
title="HMM+NP States")

g.text(gene_df.loc["Rv3461c","Gene Coef"]+0.05, gene_df.loc["Rv3461c","Gene Adj Pval"]-2.10,"Rv3461c",bbox={'facecolor': 'white', 'alpha': 0.75, 'pad': 2},ha="center",verticalalignment="bottom")
g.text(gene_df.loc["Rv2520c","Gene Coef"]+0.05, gene_df.loc["Rv2520c","Gene Adj Pval"]-2.10,"Rv2520c",bbox={'facecolor': 'white', 'alpha': 0.75, 'pad': 2},ha="center",verticalalignment="bottom")
g.text(gene_df.loc["Rv0833","Gene Coef"]+0.05, gene_df.loc["Rv0833","Gene Adj Pval"]-2.10,"Rv0833",bbox={'facecolor': 'white', 'alpha': 0.75, 'pad': 2},ha="center",verticalalignment="bottom")
g.scatter([gene_df.loc["Rv3461c","Gene Coef"],gene_df.loc["Rv2520c","Gene Coef"],gene_df.loc["Rv0833","Gene Coef"]],[gene_df.loc["Rv3461c","Gene Adj Pval"],gene_df.loc["Rv2520c","Gene Adj Pval"],gene_df.loc["Rv0833","Gene Adj Pval"]], color="none",edgecolor="black")

plt.show()


############## Model 2 Volcano #############
gene_df["States"] = gene_df["Prev States"]
gene_df.loc[gene_df["Gene+TTN Adj Pval"]< 0-np.log10(0.05), 'States'] = "Insig"
color_dict = dict({'Insig':'gray','Uncertain':'#fdc086',
                   'NE': '#386cb0',
                   'ES': '#beaed4','ESD': '#beaed4',
                   'GD': '#f0027f', 'GA':'#7fc97f'})
g=sns.scatterplot(data=gene_df, x="Gene+TTN Coef",y="Gene+TTN Adj Pval",hue="States",alpha=0.75,palette=color_dict)
g.set(xlabel="TTN Fitness Method Gene Coef", ylabel="-log10 (TTN Fitness Adjusted Pval)")
g.set_title("TTN Fitness Method")
g.legend(handles = [Line2D([0], [0], marker='o', color='w', label='NE', markerfacecolor='#386cb0', markersize=5),
Line2D([0], [0], marker='o', color='w', label='ES/ESD', markerfacecolor='#beaed4', markersize=5),
Line2D([0], [0], marker='o', color='w', label='GD', markerfacecolor='#f0027f', markersize=5),
Line2D([0], [0], marker='o', color='w', label='GA', markerfacecolor='#7fc97f', markersize=5),
Line2D([0], [0], marker='o', color='w', label='Uncertain', markerfacecolor='#fdc086', markersize=5)],
title="HMM+NP States")

g.text(gene_df.loc["Rv3461c","Gene+TTN Coef"]+0.05, gene_df.loc["Rv3461c","Gene+TTN Adj Pval"]-3.10,"Rv3461c",bbox={'facecolor': 'white', 'alpha': 0.75, 'pad': 2},ha="center",verticalalignment='bottom')
g.text(gene_df.loc["Rv2520c","Gene+TTN Coef"]+0.05, gene_df.loc["Rv2520c","Gene+TTN Adj Pval"]-3.10,"Rv2520c",bbox={'facecolor': 'white', 'alpha': 0.75, 'pad': 2},ha="center",verticalalignment="bottom")
g.text(gene_df.loc["Rv0833","Gene+TTN Coef"]+0.05, gene_df.loc["Rv0833","Gene+TTN Adj Pval"]-3.10,"Rv0833",bbox={'facecolor': 'white', 'alpha': 0.75, 'pad': 2},ha="center",verticalalignment="bottom")
g.scatter([gene_df.loc["Rv3461c","Gene+TTN Coef"],gene_df.loc["Rv2520c","Gene+TTN Coef"],gene_df.loc["Rv0833","Gene+TTN Coef"]],[gene_df.loc["Rv3461c","Gene+TTN Adj Pval"],gene_df.loc["Rv2520c","Gene+TTN Adj Pval"],gene_df.loc["Rv0833","Gene+TTN Adj Pval"]], color="none",edgecolor="black")

plt.show()

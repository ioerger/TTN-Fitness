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
import math

"""
python3 ../../Fitness_Estimation.py STLM_output H37RvBD1_hmm.csv (hmm or hmm+np) H37RvBD1.prot_table Gumbel_pred.txt > Gene_Essentiality.csv 
"""
##################################################################################
# Read in Data
ttn_data = pd.read_csv(sys.argv[1])

# HMM Stages
skip_count =0
hmm_file = open(sys.argv[2],'r')
for line in hmm_file.readlines():
	if line.startswith('#'):skip_count = skip_count+1
	else:break
hmm_file.close()
 
#specifically HMM+NP file 

hmm_stages = pd.read_csv(sys.argv[2],sep=',', skiprows=skip_count,names=["ORF ID","Name","Description","Number of TA Sites","Number of Permissive (P) Sites","Number of Non-Permissive (NP) Sites","Number of Sites Belonging to Essential State","Number of Sites Belonging to Growth-Defect State","Number of Sites Belonging to Non-Essential State","Number of Sites Belonging to Growth-Advantage State","Fraction of Sites with Insertions","Mean Normalized Read-Count At Non-Zero Sites","Final Call"])

# Prot Table
prot_table = pd.read_csv(sys.argv[3],sep='\t',header=None, names= ["Description", "X1","X2","X3","X4","X5","X6","ORF Name","ORF ID","X9","X10"])
prot_table.set_index("ORF ID", drop=True, inplace=True)

# Gumbel Predictions
skip_count =0
gumbel_file = open(sys.argv[4],'r')
for line in gumbel_file.readlines():
        if line.startswith('#'):skip_count = skip_count+1
        else:break
gumbel_file.close()

gumbel_pred = pd.read_csv(sys.argv[4],sep='\t',skiprows=skip_count, names =["Orf","Name","Desc","k","n","r","s","zbar","Call"],dtype = str)
###################################################################################
#Filter Loaded  Data
saturation = len(ttn_data[ttn_data["Count"]>0])/len(ttn_data)
phi = 1.0 - saturation
significant_n = math.log10(0.05)/math.log10(phi)

def gumbel_bernoulli_calls(data_df):
	calls = []
	for g in data_df["ORF ID"]:
		gene_call='U'
		sub_gumbel=gumbel_pred[gumbel_pred["Orf"]==g]
		if len(sub_gumbel)>0: gene_call = sub_gumbel["Call"].iloc[0]
		#set to ES if greater than n and all 0s
		sub_data = data_df[data_df["ORF ID"]==g]
		if len(sub_data)>significant_n and len(sub_data[sub_data["Count"]>0])==0: gene_call="EB"
		calls.append(gene_call)
	return calls
	
def calcPredictedCounts(row):
        predCount = row["Local Mean"]*math.pow(2,row["Pred LFC"])
        return predCount

ttn_data["Predicted Count"]=ttn_data.apply(calcPredictedCounts,axis=1)
ttn_data= ttn_data[ttn_data["ORF ID"]!="igr"]
ttn_data["Gumbel/Bernoulli Call"] = gumbel_bernoulli_calls(ttn_data)
filtered_ttn_data = ttn_data[ttn_data["Gumbel/Bernoulli Call"]!="E"]
filtered_ttn_data= filtered_ttn_data[filtered_ttn_data["Gumbel/Bernoulli Call"]!="EB"]
#filtered_ttn_data = filtered_ttn_data[filtered_ttn_data["State"]!="ES"]
filtered_ttn_data = filtered_ttn_data.reset_index(drop=True)

##########################################################################################
#Linear Regression
gene_one_hot_encoded= pd.get_dummies(filtered_ttn_data["ORF ID"],prefix='')
ttn_vectors = filtered_ttn_data.drop(["Pred LFC","Corrected Pred LFC","Corrected LFC","Predicted Count","Coord","Count","ORF ID","ORF Name","Local Mean","LFC","State", "Gumbel/Bernoulli Call"],axis=1)

X1 = pd.concat([gene_one_hot_encoded],axis=1)
X1 = sm.add_constant(X1)
X2 = pd.concat([gene_one_hot_encoded,ttn_vectors],axis=1)
X2 = sm.add_constant(X2)
Y = np.log10(filtered_ttn_data["Count"]+0.5)
results1 = sm.OLS(Y,X1).fit()
results2 = sm.OLS(Y,X2).fit()
##########################################################################################
#create Models Summary df 
def calcpval(row):
        c1 = row["M0 Coef"]
        c2 = row["M1 Coef"]
        g = row.name.split("_",1)[1]
        n = len(filtered_ttn_data[filtered_ttn_data["ORF ID"]==g])
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

Models_df["Gene+TTN States"] = "Uncertain"
Models_df.loc[(Models_df["M1 Coef"]>0) & (Models_df["M1 Adjusted Pval"]<0.05),"Gene+TTN States"]="GA"
Models_df.loc[(Models_df["M1 Coef"]<0) & (Models_df["M1 Adjusted Pval"]<0.05),"Gene+TTN States"]="GD"
Models_df.loc[(Models_df["M1 Coef"]==0) & (Models_df["M1 Adjusted Pval"]<0.05),"Gene+TTN States"]="NE"
Models_df.loc[(Models_df["M1 Adjusted Pval"]>0.05),"Gene+TTN States"]="NE"


#########################################################################################
#Write Models Information to CSV
# Columns: ORF ID, ORF Name, ORF Description,M0 Coef, M0 Adj Pval
gene_dict={} #dictionary to map information per gene
for g in ttn_data["ORF ID"].unique():
	if g == "igr": continue
	#ORF Name
	orfName = prot_table.loc[g,"ORF Name"]
	#ORF Description
	orfDescription = prot_table.loc[g,"Description"]
	#Total TA sites
	numTAsites = len(ttn_data[ttn_data["ORF ID"]==g])
	#Sites > 0
	above0TAsites = len(ttn_data[(ttn_data["ORF ID"]==g) & (ttn_data["Count"]>0)])
	#Predicted Count
	if g not in ttn_data["ORF ID"].values or ttn_data[ttn_data["ORF ID"]==g]["State"].iloc[0]=="ES": 
		pred_counts=None
		mean_pred_counts = None
	else: 
		pred_counts = filtered_ttn_data[filtered_ttn_data["ORF ID"]==g]["Predicted Count"]
		mean_pred_counts = np.mean(pred_counts)
	#Actual Count
	actual_counts = ttn_data[ttn_data["ORF ID"]==g]["Count"]
	mean_actual_counts = np.mean(actual_counts)
	#Gumbel-Bernoulli Call
	gumbel_call = gumbel_pred[gumbel_pred["Orf"]==g]["Call"].iloc[0]
	#HMM Call
	if g in hmm_stages["ORF ID"].values: hmm_call = hmm_stages[hmm_stages["ORF ID"]==g]["Final Call"].iloc[0]
	else: hmm_call = "Uncertain"
	#M0/M1 info
	if "_"+g in Models_df.index:
		used=True
		M0_coef= Models_df.loc["_"+g,"M0 Coef"]
		M0_adj_pval = Models_df.loc["_"+g,"M0 Adjusted Pval"]
		M1_coef = Models_df.loc["_"+g,"M1 Coef"]
		M1_adj_pval = Models_df.loc["_"+g,"M1 Adjusted Pval"]
		coef_diff = Models_df.loc["_"+g,"Coef Diff (M1-M0)"] 
		coef_diff_pval = Models_df.loc["_"+g,"Coef Diff Adjusted Pval"]
	else:
		used=False
		M0_coef= None
		M0_adj_pval = None
		M1_coef = None
		M1_adj_pval = None
		coef_diff = None 
		coef_diff_pval = None
	#Gene+TTN States
	gumbel_bernoulli_call = ttn_data[ttn_data["ORF ID"]==g]["Gumbel/Bernoulli Call"].iloc[0]	
	if gumbel_bernoulli_call=="E": gene_ttn_call = "ES"
	elif gumbel_bernoulli_call=="EB": gene_ttn_call = "ESB"
	else:
		if "_"+g in Models_df.index: gene_ttn_call = Models_df.loc["_"+g,"Gene+TTN States"]
		else: 
			print(g)
			gene_ttn_call = "Uncertain"

	gene_dict[g] = [g,orfName,orfDescription,numTAsites,above0TAsites,used,M0_coef,M0_adj_pval,M1_coef,M1_adj_pval,coef_diff,coef_diff_pval,mean_pred_counts,mean_actual_counts,gumbel_call,hmm_call,gene_ttn_call]

gene_df = pd.DataFrame.from_dict(gene_dict,orient='index')
gene_df.columns=["ORF ID","Name","Description","Total # TA Sites","#Sites with insertions","Used in Models","Gene (M0) Coef","Gene (M0) Adj Pval","Gene+TTN (M1) Coef","Gene+TTN (M1) Adj Pval","Coef Diff (M1-M0)","Coef Diff Adj Pval","Mean STLM Predicted Count","Mean Actual Count","Gumbel Call", "HMM+NP States","Gene+TTN States"]
#print(gene_df)


print("#Command: python3 Fitness_Esimation.py "+sys.argv[1]+" "+sys.argv[2]+" "+sys.argv[3]+" "+sys.argv[4])
print("#Gumbel Calls: Calls from the Gumbel Analysis.")
print("#Significant size n for genes lacking insertion: "+ str(significant_n))

print("#HMM States: HMM+NP model trained on data")
print("#Gene+TTN States: Genes labeled E by the Gumbel are classified as ES and those found to be essential through Bernoulli  are classified as ESB")
print("#Gene+TTN Summary: " + str(len(gene_df[gene_df["Gene+TTN States"]=="ES"]))+"ES "+ str(len(gene_df[gene_df["Gene+TTN States"]=="ESB"]))+"ESB "+ str(len(gene_df[gene_df["Gene+TTN States"]=="GD"]))+"GD "+ str(len(gene_df[gene_df["Gene+TTN States"]=="GA"]))+"GA "+ str(len(gene_df[gene_df["Gene+TTN States"]=="NE"]))+"NE" )


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
g.set(xlabel="Gene+TTN Model Gene Coef", ylabel="-log10 (Gene+TTN Model Adjusted Pval)",xlim=(-2.5,1.5),ylim=(0,100))
g.set_title("Gene+TTN Model")
g.legend(handles = [Line2D([0], [0], marker='o', color='w', label='NE', markerfacecolor='#386cb0', markersize=5),
Line2D([0], [0], marker='o', color='w', label='ES/ESD', markerfacecolor='#beaed4', markersize=5),
Line2D([0], [0], marker='o', color='w', label='GD', markerfacecolor='#f0027f', markersize=5),
Line2D([0], [0], marker='o', color='w', label='GA', markerfacecolor='#7fc97f', markersize=5),
Line2D([0], [0], marker='o', color='w', label='Uncertain', markerfacecolor='#fdc086', markersize=5)],
title="HMM+NP States")
g.axhline(0-np.log10(0.05), ls='--',color="black",alpha=0.4)
g.axvline(0,lw=3,color="black",alpha=0.4)
g.text(gene_df.loc["Rv3461c","Gene+TTN (M1) Coef"]+0.03, 0-np.log10(gene_df.loc["Rv3461c","Gene+TTN (M1) Adj Pval"])-0.025,"Rv3461c",bbox={'facecolor': 'white', 'alpha': 0.75, 'pad': 2},horizontalalignment='left')
g.text(gene_df.loc["Rv0833","Gene+TTN (M1) Coef"]+0.03, 0-np.log10(gene_df.loc["Rv0833","Gene+TTN (M1) Adj Pval"])-0.025,"Rv0833",bbox={'facecolor': 'white', 'alpha': 0.75, 'pad': 2},horizontalalignment="left")
g.scatter([gene_df.loc["Rv3461c","Gene+TTN (M1) Coef"],gene_df.loc["Rv0833","Gene+TTN (M1) Coef"]],[0-np.log10(gene_df.loc["Rv3461c","Gene+TTN (M1) Adj Pval"]),0-np.log10(gene_df.loc["Rv0833","Gene+TTN (M1) Adj Pval"])], color="none",edgecolor="black")

#plt.show()

gene_df.loc[gene_df["HMM+NP States"]=="ESD","HMM+NP States"] = "ES/ESD"
gene_df.loc[gene_df["HMM+NP States"]=="ES","HMM+NP States"] = "ES/ESD"
gene_df.loc[gene_df["Gene+TTN States"]=="ES","Gene+TTN States"] = "ES/ESD"
gene_df.loc[gene_df["Gene+TTN States"]=="ESB","Gene+TTN States"] = "ES/ESD"

prev_states = gene_df["HMM+NP States"]
new_states = gene_df["Gene+TTN States"]

cat = ["GA","NE","GD","ES/ESD","Uncertain"] 
conf_matrix = confusion_matrix(prev_states, new_states,labels=cat)
conf_matrix = pd.DataFrame(conf_matrix, index=cat, columns=cat)
plt.figure()
sns.heatmap(conf_matrix, linewidths=1, linecolor="black",annot=True, fmt='g',vmin=3000,cmap="Greys",annot_kws={"fontsize":12}).set(ylabel="HMM+NP States",xlabel="Gene+TTN States")


states = gene_df[["HMM+NP States","Gene+TTN States"]]
plt.figure()
g = sns.countplot(data=pd.melt(states), x = "value", hue="variable",alpha=0.75)
g.set_title("Number of Genes per Classification for the HMM+NP and TTN-Fitness methods")
plt.show()

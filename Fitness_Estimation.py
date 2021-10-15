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
python3 Fitness_Estimation.py STLM_output H37RvBD1.prot_table Gumbel_pred.txt > Gene_Essentiality.csv 
"""
##################################################################################
# Read in Data
skip_count =0
ttn_data = open(sys.argv[1],'r')
for line in ttn_data.readlines():
        if line.startswith('#'):skip_count = skip_count+1
        else:break
ttn_data.close()

ttn_data = pd.read_csv(sys.argv[1],skiprows=skip_count)

# Prot Table
prot_table = pd.read_csv(sys.argv[2],sep='\t',header=None, names= ["Description", "X0","X1","X3","X4","X5","X6","ORF Name","ORF ID","X9","X00"])
prot_table.set_index("ORF ID", drop=True, inplace=True)

# Gumbel Predictions
skip_count =0
gumbel_file = open(sys.argv[3],'r')
for line in gumbel_file.readlines():
        if line.startswith('#'):skip_count = skip_count+1
        else:break
gumbel_file.close()

gumbel_pred = pd.read_csv(sys.argv[3],sep='\t',skiprows=skip_count, names =["Orf","Name","Desc","k","n","r","s","zbar","Call"],dtype = str)

###################################################################################
#Filter Loaded  Data
saturation = len(ttn_data[ttn_data["Count"]>0])/len(ttn_data)
phi = 1.0 - saturation
significant_n = math.log10(0.05)/math.log10(phi)

def gumbel_binomial_calls(data_df):
	calls = []
	for g in data_df["ORF ID"].unique():
		gene_call='U'
		sub_gumbel=gumbel_pred[gumbel_pred["Orf"]==g]
		if len(sub_gumbel)>0: gene_call = sub_gumbel["Call"].iloc[0]
		#set to ES if greater than n and all 0s
		sub_data = data_df[data_df["ORF ID"]==g]
		if len(sub_data)>significant_n and len(sub_data[sub_data["Count"]>0])==0: gene_call="EB"
		calls.extend([gene_call]*len(data_df[data_df["ORF ID"]==g]))
	return calls
	
def calcPredictedCounts(row):
        predCount = row["Local Mean"]*math.pow(2,row["Pred LFC"])
        return predCount

ttn_data["Predicted Count"]=ttn_data.apply(calcPredictedCounts,axis=1)
ttn_data= ttn_data[ttn_data["ORF ID"]!="igr"]
ttn_data["Gumbel/Binomial Call"] = gumbel_bernoulli_calls(ttn_data)
filtered_ttn_data = ttn_data[ttn_data["Gumbel/Binomial Call"]!="E"]
filtered_ttn_data= filtered_ttn_data[filtered_ttn_data["Gumbel/Binomial Call"]!="EB"]
filtered_ttn_data = filtered_ttn_data.reset_index(drop=True)

##########################################################################################
#Linear Regression
gene_one_hot_encoded= pd.get_dummies(filtered_ttn_data["ORF ID"],prefix='')
ttn_vectors = filtered_ttn_data.drop(["Pred LFC","Corrected Pred LFC","Corrected LFC","Predicted Count","Coord","Count","ORF ID","ORF Name","Local Mean","LFC","State", "Gumbel/Binomial Call"],axis=1)

X0 = pd.concat([gene_one_hot_encoded],axis=1)
X0 = sm.add_constant(X0)

X1 = pd.concat([gene_one_hot_encoded,ttn_vectors],axis=1)
X1 = sm.add_constant(X1)
Y = np.log10(filtered_ttn_data["Count"]+0.5)

results0 = sm.OLS(Y,X0).fit()
results1 = sm.OLS(Y,X1).fit()

filtered_ttn_data["M0 Pred log Count"] = results0.predict(X0)
filtered_ttn_data["M1 Pred log Count"] = results1.predict(X1)

def calcPredictedCounts(row):
	predM0Count = math.pow(10,row["M0 Pred log Count"])-0.5
	predM1Count = math.pow(10,row["M1 Pred log Count"])-0.5
	return [predM0Count,predM1Count]

filtered_ttn_data[["M0 Predicted Count","M1 Predicted Count"]]=filtered_ttn_data.apply(calcPredictedCounts,axis=1,result_type="expand")
##########################################################################################
#create Models Summary df 
Models_df = pd.DataFrame(results0.params[1:],columns=["M0 Coef"])
Models_df["M0 Pval"] = results0.pvalues[1:]
Models_df["M0 Adjusted Pval"] = statsmodels.stats.multitest.fdrcorrection(results0.pvalues[1:],alpha=0.05)[1]
Models_df["M1 Coef"]= results1.params[1:-256]
Models_df["M1 Pval"] = results1.pvalues[1:-256]
Models_df["M1 Adjusted Pval"] = statsmodels.stats.multitest.fdrcorrection(results1.pvalues[1:-256],alpha=0.05)[1]
#creating a mask for the adjusted pvals
Models_df.loc[(Models_df["M1 Coef"]>0) & (Models_df["M1 Adjusted Pval"]<0.05),"Gene+TTN States"]="GA"
Models_df.loc[(Models_df["M1 Coef"]<0) & (Models_df["M1 Adjusted Pval"]<0.05),"Gene+TTN States"]="GD"
Models_df.loc[(Models_df["M1 Coef"]==0) & (Models_df["M1 Adjusted Pval"]<0.05),"Gene+TTN States"]="NE"
Models_df.loc[(Models_df["M1 Adjusted Pval"]>0.05),"Gene+TTN States"]="NE"


#########################################################################################
#Write Models Information to CSV
# Columns: ORF ID, ORF Name, ORF Description,M0 Coef, M0 Adj Pval
gene_dict={} #dictionary to map information per gene
for g in ttn_data["ORF ID"].unique():
	#ORF Name
	orfName = prot_table.loc[g,"ORF Name"]
	#ORF Description
	orfDescription = prot_table.loc[g,"Description"]
	#Total TA sites
	numTAsites = len(ttn_data[ttn_data["ORF ID"]==g])
	#Sites > 0
	above0TAsites = len(ttn_data[(ttn_data["ORF ID"]==g) & (ttn_data["Count"]>0)])
	#Actual Count
	actual_counts = ttn_data[ttn_data["ORF ID"]==g]["Count"]
	mean_actual_counts = np.mean(actual_counts)
	#Predicted Count
	if g not in ttn_data["ORF ID"].values or ttn_data[ttn_data["ORF ID"]==g]["State"].iloc[0]=="ES": 
		M0_ratio=None
		M1_ratio = None
	else: 
		M0_ratio = np.log10(np.mean(filtered_ttn_data[filtered_ttn_data["ORF ID"]==g]["M0 Predicted Count"])/mean_actual_counts)
		M1_ratio = np.log10(np.mean(filtered_ttn_data[filtered_ttn_data["ORF ID"]==g]["M0 Predicted Count"])/mean_actual_counts)
	#M0/M1 info
	if "_"+g in Models_df.index:
		used=True
		M0_coef= Models_df.loc["_"+g,"M0 Coef"]
		M0_adj_pval = Models_df.loc["_"+g,"M0 Adjusted Pval"]
		M1_coef = Models_df.loc["_"+g,"M1 Coef"]
		M1_adj_pval = Models_df.loc["_"+g,"M1 Adjusted Pval"]
		
	else:
		used=False
		M0_coef= None
		M0_adj_pval = None
		M1_coef = None
		M1_adj_pval = None
	#States
	gumbel_bernoulli_call = ttn_data[ttn_data["ORF ID"]==g]["Gumbel/Binomial Call"].iloc[0]	
	if gumbel_bernoulli_call=="E": gene_ttn_call = "ES"
	elif gumbel_bernoulli_call=="EB": gene_ttn_call = "ESB"
	else:
		if "_"+g in Models_df.index: gene_ttn_call = Models_df.loc["_"+g,"Gene+TTN States"]
		else: gene_ttn_call = "Uncertain"
	gene_dict[g] = [g,orfName,orfDescription,numTAsites,above0TAsites,used,M0_coef,M0_adj_pval,M1_coef,M1_adj_pval,M0_ratio,M1_ratio,mean_actual_counts,gene_ttn_call]

gene_df = pd.DataFrame.from_dict(gene_dict,orient='index')
gene_df.columns=["ORF ID","Name","Description","Total # TA Sites","#Sites with insertions","Used in Models","Gene (M0) Coef","Gene (M0) Adj Pval","Gene+TTN (M1) Coef","Gene+TTN (M1) Adj Pval","M0 Fitness Estimation","M1 Fitness Estimation","Mean Actual Count", "TTN-Fitness Assesment"]
#print(gene_df)


print("#Command: python3 Fitness_Esimation.py "+sys.argv[1]+" "+sys.argv[2]+" "+sys.argv[3])
print("#Gumbel/Bernoulli Calls: Calls from the Gumbel Analysis. Genes found to be significant through Bernoulli when all sites have 0 insertions are labeled EB")
print("#Significant size n for genes lacking insertion: "+ str(significant_n))

print("#HMM States: HMM+NP model trained on data")
print("#Gene+TTN States: Genes labeled E by the Gumbel are determined to be ES and those labeled EB by Gumbel are determined to be ESB")
print("#Gene+TTN Summary: " + str(len(gene_df[gene_df["TTN-Fitness Assesment"]=="ES"]))+"ES "+ str(len(gene_df[gene_df["TTN-Fitness Assesment"]=="ESB"]))+"ESB "+ str(len(gene_df[gene_df["TTN-Fitness Assesment"]=="GD"]))+"GD "+ str(len(gene_df[gene_df["TTN-Fitness Assesment"]=="GA"]))+"GA "+ str(len(gene_df[gene_df["TTN-Fitness Assesment"]=="NE"]))+"NE" )


gene_data = gene_df.to_csv(header=True, index=False).split('\n')
vals = '\n'.join(gene_data)
print(vals)


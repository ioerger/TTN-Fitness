import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import itertools
import scipy.stats as stats
'''
python3 Heatmaps.py LFCs.txt combinedwig > wigCorrelation.csv
1. If combined wig is passed in, create a heatmap of the correlation of the wig files
2. Preprocess LFC data to exclude TA sites marked essential and expand the nuclotides within 20 bps from the TA site
3. Line Graph of the overall probability of nucleotides in each position
4. Line Graphs of probability of nucleotides in the high middle and lower third ranges of insertion counts
'''
###################################################################################################################
# if combined wigs file provided, then create the heatmap + output
if (len(sys.argv)>2):
	combined_wig = pd.read_csv(sys.argv[2],header=None,sep='\t',skiprows=17).iloc[:,1:15]
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


##############################################################################################################################
# LFC data process
LFC_data = pd.read_csv(sys.argv[1],sep="\t",header=None,names=["Coord","ORF ID","ORF Name","Nucl Window","State","Count","Local Mean","LFC","Description"])
sample_name = sys.argv[1].replace('_LFCs.txt','')
sample_name = sample_name.split('/')[-1]

#filter out ES
LFC_data = LFC_data[LFC_data["State"]!="ES"]

expanded_nucl_data = LFC_data["Nucl Window"].apply(lambda x: pd.Series(list(x)))
expanded_nucl_data.columns = [-20,-19,-18,-17,-16,-15,-14,-13,-12,-11,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,'T','A',1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
expanded_nucl_data["Coord"]=LFC_data["Coord"]
expanded_nucl_data["Count"]=LFC_data["Count"]
expanded_nucl_data["Local Mean"]=LFC_data["Local Mean"]
expanded_nucl_data["LFC"] = LFC_data["LFC"]


##########################################################
############  FIGURES  ###################################
##########################################################
#heatmaps of LFC average per nucleotide per position
heatmap_df = pd.DataFrame()
for col in expanded_nucl_data.columns[:-4]:
	#in the order ACTG
	average_LFC_col= []
	for nucl in ['A','T','C','G']:
		nucl_filtered =expanded_nucl_data[expanded_nucl_data[col]==nucl]
		average_LFC_col.append(nucl_filtered["LFC"].mean())
	heatmap_df[col]=average_LFC_col

heatmap_df.index=['A','T','C','G']
median = heatmap_df.median(axis=0).median()
plt.figure(figsize=(10,3))
sns.heatmap(heatmap_df,cmap='PRGn',center=median,vmin=median-1.5,vmax=median+1.5,square=True,cbar_kws={"shrink": 0.75,'label': 'Mean LFCs'}).set_title(sample_name)

#LFC Distribution Plot
plt.figure()
sns.distplot(expanded_nucl_data["LFC"],bins=50).set_title(sample_name+" LFC Distribution")
plt.show()

##### Nucleotide Probability Graphs###########
def create_count_df(input_df):
	A=[]
	G=[]
	C=[]
	T=[]
	for col in heatmap_df.columns:
		col_counts = input_df[col].value_counts()
		if "G" in col_counts.index: G.append(col_counts["G"]/len(input_df[col]))
		else: G.append(0)
		if "A" in col_counts.index: A.append(col_counts["A"]/len(input_df[col]))
		else: A.append(0)
		if "T" in col_counts.index: T.append(col_counts["T"]/len(input_df[col]))
		else: T.append(0)
		if "C" in col_counts.index: C.append(col_counts["C"]/len(input_df[col]))
		else: C.append(0)

	output_df = pd.DataFrame()
	output_df["A"]=A
	output_df["T"]=T
	output_df["G"]=G
	output_df["C"]=C
	return output_df

#Overall Prob Line Graph
prob_df = create_count_df(expanded_nucl_data)
prob_df["pos"] = range(42)
prob_df = prob_df.melt('pos',var_name='Nucleotides', value_name='Prob')

#plt.figure(figsize=(10,3))
plt.figure(figsize=(10,3))
g=sns.lineplot(x="pos", y="Prob", hue="Nucleotides", data=prob_df)
g.set(xlabel="Position",ylabel="Probability",xticks=range(42))
g.set_xticklabels(rotation=90,labels=heatmap_df.columns)
g.set_title("Probabilty of Nucleotide Occurrence Per Position in "+sample_name)
#plt.show()

expanded_nucl_data[["Count"]]=np.log(expanded_nucl_data[['Count']].replace(0, np.nan))

max_LFC = expanded_nucl_data["Count"].max()
min_LFC = expanded_nucl_data["Count"].min()
thirds = (max_LFC-min_LFC)/3

high_third_df = expanded_nucl_data[expanded_nucl_data["Count"].between(max_LFC-thirds,max_LFC)]
med_third_df = expanded_nucl_data[expanded_nucl_data["Count"].between(min_LFC+thirds,max_LFC-thirds)]
low_third_df = expanded_nucl_data[expanded_nucl_data["Count"].between(min_LFC,min_LFC+thirds)]

#High Prob Line Graph
prob_df = create_count_df(high_third_df)
prob_df["pos"] = range(42)
prob_df = prob_df.melt('pos',var_name='Nucleotides', value_name='Prob')
plt.figure(figsize=(10,3))
g=sns.lineplot(x="pos", y="Prob", hue="Nucleotides", data=prob_df)
g.set(xlabel="Position",ylabel="Probability",xticks=range(42))
g.set_xticklabels(rotation=90,labels=heatmap_df.columns)
g.set_title("Probabilty of Nucleotide Occurrence Per Position High Third in "+sample_name)
#plt.show()

#Medium Prob Line Graph
prob_df = create_count_df(med_third_df)
prob_df["pos"] = range(42)
prob_df = prob_df.melt('pos',var_name='Nucleotides', value_name='Prob')
plt.figure(figsize=(10,3))
g=sns.lineplot(x="pos", y="Prob", hue="Nucleotides", data=prob_df)
g.set(xlabel="Position",ylabel="Probability",xticks=range(42))
g.set_xticklabels(rotation=90,labels=heatmap_df.columns)
g.set_title("Probabilty of Nucleotide Occurrence Per Position Med Third in "+sample_name)
#plt.show()

#Low Prob Line Graph
prob_df = create_count_df(low_third_df)
prob_df["pos"] = range(42)
prob_df = prob_df.melt('pos',var_name='Nucleotides', value_name='Prob')
plt.figure(figsize=(10,3))
g=sns.lineplot(x="pos", y="Prob", hue="Nucleotides", data=prob_df)
g.set(xlabel="Position",ylabel="Probability",xticks=range(42))
g.set_xticklabels(rotation=90,labels=heatmap_df.columns)
g.set_title("Probabilty of Nucleotide Occurrence Per Position Low Third in "+sample_name)
plt.show()



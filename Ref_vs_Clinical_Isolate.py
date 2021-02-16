#python3 ref_vs_clincal_isolates.py ref_ttn targettn compareTASite picklefile > ref_isolate_mean_CSV
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
import seaborn as sns

#load in regression model
with tarfile.open(sys.argv[4]+'.tar.gz', 'r') as t:
    t.extractall('')
reg = sm.load(os.path.basename(sys.argv[4]))
os.remove(os.path.basename(sys.argv[4]))

#read reference ttn data
ref_data = pd.read_csv(sys.argv[1])
ref_sample_name = sys.argv[1].replace('_tetranucl.csv','')
ref_sample_name = ref_sample_name.split('/')[-1]

#read isolate ttn data
isolate_data = pd.read_csv(sys.argv[2])
isolate_sample_name = sys.argv[2].replace('_tetranucl.csv','')
isolate_sample_name = isolate_sample_name.split('/')[-1]

#extract sites with nucl differences within TTN
compare_sites = pd.read_csv(sys.argv[3],sep=" ",header=None,names=["ref Coord","ref Nucl","isolate Coord","isolate Nucl","compare filter"])
compare_sites = compare_sites[compare_sites["compare filter"]=="*"]
compare_sites["ref Coord"] = compare_sites["ref Coord"].astype('int64')
compare_sites["isolate Coord"] = compare_sites["isolate Coord"].astype('int64')

##################################################################################
def ttn_diff(s1,s2): #find Nucl differences in TTN of the two strains at a site
	num_diff = 0
	refNucl=[]
	isoNucl=[]
	locs=[]
	for i in range(len(s1)):
		if(s1[i]!=s2[i]):
			num_diff = num_diff +1
			refNucl.append(s1[i])
			isoNucl.append(s2[i])
			locs.append(i)
	return (num_diff,refNucl,isoNucl,locs)


#extract rows with coords that have onlt only one nucleotide difference in alligned reference and isolate dataframes
nucl_ref = []
nucl_isolate = []
ref_SNP=[]
iso_SNP = []
loc_SNP =[]
ref_row_list = []
isolate_row_list=[]
for idx,row in compare_sites.iterrows():		
	ref_subset = ref_data[ref_data["Coord"] == row["ref Coord"]]
	isolate_subset = isolate_data[isolate_data["Coord"] == row["isolate Coord"]]
	if (len(ref_subset)>0 and len(isolate_subset)>0):# make sure both indicies are present in ttn list
		output=ttn_diff(row["ref Nucl"],row["isolate Nucl"])
		if(output[0]==1): #keep rows that only have one difference
			nucl_ref.append(row["ref Nucl"])
			nucl_isolate.append(row["isolate Nucl"])
			ref_row_list.append(ref_subset)
			isolate_row_list.append(isolate_subset)
			ref_SNP.append(output[1][0])
			iso_SNP.append(output[2][0])
			loc_SNP.append(output[3][0]) 
ref_filtered = pd.concat(ref_row_list)
ref_filtered = ref_filtered.reset_index(drop=True)
isolate_filtered=pd.concat(isolate_row_list) 
isolate_filtered = isolate_filtered.reset_index(drop=True)

######################################################################################################
# REGRESSION TO GET PREDICTED LFCS OF THE TTTNs
ref_ttn = ref_filtered.drop(["Coord","Count","Local Mean","LFC"],axis=1)
isolate_ttn = isolate_filtered.drop(["Coord","Count","Local Mean","LFC"],axis=1)

ref_ttn = sm.add_constant(ref_ttn)
ref_pred_LFC = reg.predict(ref_ttn)

isolate_ttn = sm.add_constant(isolate_ttn)
isolate_pred_LFC = reg.predict(isolate_ttn)

#####################################################################################################
compare_data = pd.DataFrame()
compare_data["Ref Seq"] = nucl_ref
compare_data["Isolate Seq"] = nucl_isolate
compare_data["Loc SNP"] = loc_SNP
compare_data["Ref SNP"] = ref_SNP
compare_data["Iso SNP"] = iso_SNP
compare_data["Ref Obs LFC"]= ref_filtered["LFC"]
compare_data["Isolate Obs LFC"] = isolate_filtered["LFC"]
compare_data["Ref Pred LFC"] = ref_pred_LFC
compare_data["Isolate Pred LFC"] = isolate_pred_LFC
compare_data["Obs LFC Difference"] = isolate_filtered["LFC"] - ref_filtered["LFC"]
compare_data["Pred LFC Difference"] = isolate_pred_LFC-ref_pred_LFC

######################################################################################################

#Plot Overall Diff Observed vs. Diff Predicted 
fig, (ax1) = plt.subplots(1, sharex=True, sharey=True)
fig.suptitle("Delta Observed vs. Delta predicted of: "+str(ref_sample_name)+" and "+str(isolate_sample_name))
ax1.scatter(compare_data["Obs LFC Difference"],compare_data["Pred LFC Difference"],s=5,c='green',alpha=0.75,label="original")
ax1.set_xlabel('Observed Delta Reference LFC')
ax1.set_ylabel('Predicted Delta Isolate LFC')
ax1.axhline(y=0, color='k')
ax1.axvline(x=0, color='k')
ax1.plot([-6,6], [-6,6], 'k--', alpha=0.25, zorder=1)
ax1.text(-6, 4, "Reference R2: "+ str(round(r2_score(ref_obs_LFC,ref_pred_LFC),4)), fontsize=10)
ax1.text(-6, 3.5,"Isolate R2: "+ str(round(r2_score(isolate_obs_LFC,isolate_pred_LFC),4)), fontsize=10)
ax1.grid(zorder=0)

##########################################################################################################
# Get mean obs lfc difference and mean pred lfc difference per reference Nucleotide-isolare Nucleotide-SNPlocation combo.

ref_col=[]
iso_col=[]
num_inst=[]
pos=[]
mean_obs = []
mean_pred = []
mean_diff = []


for r_idx,refNucl in enumerate(["A","T","C","G"]):
	for isoNucl in ["A","T","C","G"]:
		for i in [0,1,2,3,6,7,8,9]:
			if refNucl == isoNucl : continue
			ref_col.append(refNucl)
			iso_col.append(isoNucl)
			pos.append(i)
			subsetDF = compare_data[(compare_data["Ref SNP"]==refNucl) & (compare_data["Iso SNP"]==isoNucl) & (compare_data["Loc SNP"]==i)]
			num_inst.append(len(subsetDF))
			mean_obs.append(subsetDF["Obs LFC Difference"].mean())
			mean_pred.append(subsetDF["Pred LFC Difference"].mean())
                        
nucl_combo_mean_df = pd.DataFrame([ref_col,iso_col,pos,num_inst,mean_obs,mean_pred])
nucl_combo_mean_df = nucl_combo_mean_df.transpose()
nucl_combo_mean_df.columns=["Ref Nucl","Iso Nucl","Position","Num Instances","Mean Delta Obs","Mean Delta Pred"] 
nucl_combo_mean_df["Position Cat"] = nucl_combo_mean_df["Position"].replace( [ 0,1,2,3,6,7,8,9 ],[ "+4/-4","+3/-3","+2/-2","+1/-1","+1/-1","+2/-2","+3/-3","+4/-4"])
nucl_combo_mean_df["Position"] = nucl_combo_mean_df["Position"].replace( [ 0,1,2,3,6,7,8,9 ],[ "-4","-3","-2","-1","+1","+2","+3","+4"])

nucl_combo_mean_df = nucl_combo_mean_df.to_csv(header=True, index=False).split('\n')
vals = '\n'.join(nucl_combo_mean_df)
print(vals)

#########################################################################################################
# Most siginificant nucleotide biases
A3 = nucl_combo_mean_df[(nucl_combo_mean_df["Ref Nucl"]=="A") & (nucl_combo_mean_df["Position"]=="-3")]
G2 = nucl_combo_mean_df[(nucl_combo_mean_df["Ref Nucl"]=="G") & (nucl_combo_mean_df["Position"]=="-2")]
T3 = nucl_combo_mean_df[(nucl_combo_mean_df["Ref Nucl"]=="T") & (nucl_combo_mean_df["Position"]=="+3")]
C2 = nucl_combo_mean_df[(nucl_combo_mean_df["Ref Nucl"]=="C") & (nucl_combo_mean_df["Position"]=="+2")]


#Plot obs vs. pred diff per SNP and position pair                
scatterfig,ax1 = plt.subplots(1)
scatterfig.suptitle("Mean Delta Observed vs. Mean Predicted LFC per SNP")
ax1.scatter(nucl_combo_mean_df["Mean Delta Obs"],nucl_combo_mean_df["Mean Delta Pred"],c="gray",s=10)
ax1.scatter(A3["Mean Delta Obs"],A3["Mean Delta Pred"], c= "dodgerblue", label="A-3N",s=15)
ax1.scatter(G2["Mean Delta Obs"],G2["Mean Delta Pred"], c="orange", label="G-2N",s=15)
ax1.scatter(T3["Mean Delta Obs"],T3["Mean Delta Pred"], c="green", label= "T+3N",s=15)
ax1.scatter(C2["Mean Delta Obs"],C2["Mean Delta Pred"], c="fuchsia", label="C+2N",s=15)
ax1.set_xlabel("Mean Delta Observed LFC")
ax1.set_ylabel("Mean Delta Predicted LFC")
ax1.set_xlim(-2.5,2.5)
ax1.set_ylim(-2.5,2.5)
plt.legend()
plt.show()

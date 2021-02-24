import sys
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import seaborn as sns

"""
python3 LFCs.txt > TTN.csv
1. Preprocess LFC data to exclude the sites marked essential and expand the nucleotides surrounding the TA site
2. reverse complement the upstream sequence
3. Combine the downstream and RC upstream 
4. For every row, create the 256-bit vector setting the upstream and downtream TTN
5. Write out to file
"""
##########################################################################################################
# Preprocess LFC data

LFC_data = pd.read_csv(sys.argv[1],sep="\t",header=None)
LFC_data.columns= ["Coord","ORF ID","ORF Name","Nucl Window","State","Count","Local Mean","LFC","Description"]

#filter out ES
LFC_data = LFC_data[LFC_data["State"]!="ES"]
LFC_data = LFC_data.reset_index(drop=True)

#expand out the nucleotides into their own columns
expanded_nucl = LFC_data["Nucl Window"].apply(lambda x: pd.Series(list(x)))
expanded_nucl.columns = [-20,-19,-18,-17,-16,-15,-14,-13,-12,-11,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,'T','A',1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]

# Reverse Complement the upstream nucleotides
#complement the nucleotides functions
def complement(x):
	if(str(x)=="A"): return "T"
	if(str(x)=="C"): return "G"
	if(str(x)=="G"): return "C"
	if(str(x)=="T"): return "A"

rev_complement_columns = list(expanded_nucl.columns)[22:43]
rev_complement_columns.reverse()
rev_df = expanded_nucl[rev_complement_columns].applymap(lambda x: complement(x))
orig_df = expanded_nucl[list(expanded_nucl.columns)[0:22]]
up_rc_down_df = pd.concat([orig_df, rev_df], axis=1)
up_rc_down_df.columns = [-20,-19,-18,-17,-16,-15,-14,-13,-12,-11,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,'T','A',1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]

###########################################################################################################
#function to create the TTN bit vector
combos=[''.join(p) for p in itertools.product(['A','C','T','G'], repeat=4)]
def generate_bit_vector(row):
	row_list = []
	for c in combos:
		if row["Upseq"]==c and row["Downseq"]==c: row_list.append(int(2)) #up/dwn ttn are same, "bit"=2
		elif row["Upseq"]==c or row["Downseq"]==c:row_list.append(int(1)) #set ttn bit=1 
		else:row_list.append(int(0))
	return pd.Series(row_list,index=combos)


temp_df = pd.DataFrame()
temp_df["Upseq"]=up_rc_down_df[-4]+up_rc_down_df[-3]+up_rc_down_df[-2]+up_rc_down_df[-1]
temp_df["Downseq"] = up_rc_down_df[17]+up_rc_down_df[18]+up_rc_down_df[19]+up_rc_down_df[20]

#create te dataframe and write to out
tetra_nucl_data = temp_df.apply(generate_bit_vector,axis=1,result_type="expand")#get TTN vectors for every site
tetra_nucl_data["Coord"] = LFC_data["Coord"]
tetra_nucl_data["Count"] = LFC_data["Count"]
tetra_nucl_data["Local Mean"] = LFC_data["Local Mean"]
tetra_nucl_data["LFC"] = LFC_data["LFC"]

tetra_nucl_data = tetra_nucl_data.to_csv(header=True, index=False).split('\n')
vals = '\n'.join(tetra_nucl_data)
print(vals)

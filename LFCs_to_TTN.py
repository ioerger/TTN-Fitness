import sys
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import seaborn as sns

#input is the diffs file computed from computeLFCs.py
#output is OneHotEncoded Nucleotides
#Functionality: Filter out the NE genes,RC(Downstream) and create 256 bit vector per sample
#command: python3 diffsFile > OneHotEncoded.csv


LFC_data = pd.read_csv(sys.argv[1],sep="\t",header=None)
LFC_data.columns= ["Coord","NuclWindow","State","Count","Local Mean","LFC"]

#filter out ES
LFC_data = LFC_data[LFC_data["State"]!="ES"]
LFC_data = LFC_data.reset_index(drop=True)

#split nucl is own columns
def strip_list(mylist):
    out_list=[]
    for element in mylist:
        new_element = element.rstrip().lstrip()
        if new_element!="":
            out_list.append(new_element)
    return out_list

expanded_nucl= LFC_data.NuclWindow.astype(str)
expanded_nucl= expanded_nucl.str.split('([ATCG])', expand = False).apply(strip_list)
expanded_nucl = expanded_nucl.apply(pd.Series)
expanded_nucl.columns=expanded_nucl.columns.astype(str)

#reverse-complement downstream
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
up_rc_down_df.columns = [-20,-19,-18,-17,-16,-15,-14,-13,-12,-11,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,
                      'T','A',
                      1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
#create 256 bit vector per sample
combos=[''.join(p) for p in itertools.product(['A','C','T','G'], repeat=4)]
def generate_bit_vector(row):
	
	row_list = []
	for c in combos:
		if row["Upseq"]==c and row["Downseq"]==c:
			row_list.append(int(2))
		elif row["Upseq"]==c or row["Downseq"]==c:
			row_list.append(int(1))
		else:
			row_list.append(int(0))
	return pd.Series(row_list,index=combos)


temp_df = pd.DataFrame()
temp_df["Upseq"]=up_rc_down_df[-4]+up_rc_down_df[-3]+up_rc_down_df[-2]+up_rc_down_df[-1]
temp_df["Downseq"] = up_rc_down_df[17]+up_rc_down_df[18]+up_rc_down_df[19]+up_rc_down_df[20]

tetra_nucl_data = temp_df.apply(generate_bit_vector,axis=1,result_type="expand")#expansion of dictionary results in strange addition of columns thus have to drop NA
tetra_nucl_data["Coord"] = LFC_data["Coord"]
tetra_nucl_data["Count"] = LFC_data["Count"]
tetra_nucl_data["Local Mean"] = LFC_data["Local Mean"]
tetra_nucl_data["LFC"] = LFC_data["LFC"]

tetra_nucl_data = tetra_nucl_data.to_csv(header=True, index=False).split('\n')
vals = '\n'.join(tetra_nucl_data)
print(vals)

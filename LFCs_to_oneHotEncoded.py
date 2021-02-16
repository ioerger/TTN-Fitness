import sys
import pandas as pd


#input is the diffs file computed from computeLFCs.py
#output is OneHotEncoded Nucleotides
#Functionality: Filter out the NE genes. OneHotEncode Nucleotides and write to file. As is, no reverse complementing performed.
#command: python3 diffsFile > OneHotEncoded.csv
#reverse complement the upstream ones
def complement(x):
    if(str(x)=="A"): return "T"
    if(str(x)=="C"): return "G"
    if(str(x)=="G"): return "C"
    if(str(x)=="T"): return "A"
 
#read in LFC dataframe
LFC_data = pd.read_csv(sys.argv[1],sep="\t",header=None)
LFC_data.columns= ["Coord","Orf ID","Orf Name","Nucl Window","State","Count","Local Mean","LFC", "Description"]
sample_name = sys.argv[1].replace('_LFCs.txt','')
sample_name = sample_name.split('/')[-1]

#filter out ES
LFC_data = LFC_data[LFC_data["State"]!="ES"]

expanded_data = LFC_data["Nucl Window"].apply(lambda x: pd.Series(list(x)))
expanded_data.columns = [-20,-19,-18,-17,-16,-15,-14,-13,-12,-11,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,'T','A',1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]

#rerse complement downstream
rev_complement_columns = list(expanded_data.columns)[22:43]
rev_complement_columns.reverse()
rev_df = expanded_data[rev_complement_columns].applymap(lambda x: complement(x))
orig_df = expanded_data[list(expanded_data.columns)[0:22]]
expanded_data = pd.concat([orig_df, rev_df], axis=1)

#one-hot-encoded
expanded_data = pd.get_dummies(data=expanded_data, columns=expanded_data.columns)
expanded_data["Coord"]=LFC_data["Coord"]
expanded_data["Count"]=LFC_data["Count"]
expanded_data["Local Mean"]=LFC_data["Local Mean"]
expanded_data["LFC"] = LFC_data["LFC"]


expanded_data = expanded_data.to_csv(header=True,index=False).split('\n')
vals = '\n'.join(expanded_data)
print(vals)

"""
Command: python333 compute_LFCs.py genome.fna genome.prot_table <list of counts.wig files> > genome_LFCs.txt

1. load in the fna file as one long string of characters, load in genes description from the prottable, read in wig files and combine them into parallel array matching Coords and corresponding counts in the wig files
2. Get average normalized counts per TA site from wig files
3. Get nucleotides -20bps and +20bps from the TAsites.
4. Create priliminary essentiality labels for the TA sites (ES = sites of count 0 of 6 more consecutive TA sites)
5. Compute Local Means per TA site (Mean of past 5 and next 5 TA sites, exluding self)
6. Using the Local Means and Counts at every TA site to find the LFCs
"""
import sys,math,numpy

#read in the fna file as one continous string
def read_genome(filename):
  s = ""
  n = 0
  for line in open(filename):
    if n==0: n = 1 # skip first
    else: s += line[:-1]
  return s

#read in postion and counts of insertions
def read_wig(fname):
  coords,counts = [],[]
  for line in open(fname):
    if line[0] not in "123456789": continue
    w = line.split()
    pos,cnt = int(w[0]),float(w[1])
    coords.append(pos)
    counts.append(cnt)
  return coords,counts

#read in values from prot table
def read_genes(fname,descriptions=False):
  genes = []
  for line in open(fname):
    w = line.split('\t')
    data = [int(w[1])-1,int(w[2])-1,w[8],w[7],w[3]]
    if descriptions==True: data.append(w[0])
    genes.append(data)
  return genes

def hash_genes(genes):
  hash = {}
  for gene in genes:
    a,b = gene[0],gene[1]
    for i in range(a,b+1):
      hash[i] = gene
  return hash

g = read_genome(sys.argv[1])
N = len(g)
genes = read_genes(sys.argv[2],descriptions=True)
ghash = hash_genes(genes)
 
# read in wig files
coords,allcounts = [],[]
for wig in sys.argv[3:]:
  co,cnts = read_wig(wig)
  if coords==[]: coords = co
  elif len(co)!=len(coords): sys.stderr.write("error: wig files have different number of sites"); sys.exit(0)
  allcounts.append(cnts)

# normalize counts within each dataset, and compute mean at each TA site
temp = numpy.array(allcounts).transpose()
means = temp.mean(axis=0)
temp = 100.0*temp/means # normalized so mean cnt=100 for each dataset, but ignores outliers or diffs in saturation
counts = temp.transpose().mean(axis=0) 

# extract nucleotides in window around each TA site

g2 = g+g # wrap-around for TA sites near termini
nucs = []
for co in coords:
  co -= 1 # 1-based to 0-based indexing of nucleotides
  if co-20<0: co += N
  nucs.append(g2[co-20:co+22])
  if nucs[-1][20:22]!="TA": sys.stderr.write("warning: site %d is %s instead of TA" % (co,nucs[-1][20:22]))

# compute state labels (ES or NE)
# for runs of >=R TA sites with cnt=0; label them as "ES", and the rest as "NE"
# treat ends of genome as connected (circular)

Nsites = len(counts)
states = ["NE"]*Nsites
R = 6 # make this adaptive based on saturation?
MinCount = 2
i = 0
while i<Nsites:
  j = i
  while j<Nsites and counts[j]<MinCount: j += 1
  if j-i>=R:
    for k in range(i,j): states[k] = "ES"
    i = j
  else: i += 1

# compute local means (exclude self)

W = 5
localmeans = []
for i in range(Nsites):
  vals = []
  for j in range(-W,W+1):
    #if i+j>=0 and i+j<Nsites: # this includes the site itself (diffs3.txt)
    if j!=0 and i+j>=0 and i+j<Nsites: # this excludes the site itself 
      if states[i+j]!=states[i]: continue # include only neighboring sites with same state when calculating localmean # diffs2.txt !!!
      vals.append(float(counts[i+j]))
  smoothed = -1 if len(vals)==0 else numpy.mean(vals)
  #if states[i]!="NE": smoothed = -1
  localmeans.append(smoothed)

# calc LFCs and print out 

PC = 10
for i in range(Nsites):
  c,m = counts[i],localmeans[i]
  gene = ghash.get(coords[i],None)
  orfid = "igr" if gene==None else "%s"%(gene[2])
  orfName = "igr" if gene==None else "%s"%(gene[3])
  descr = "igr" if gene==None else "%s"%(gene[5])
  lfc = math.log((c+PC)/float(m+PC),2)
  vals = [coords[i],orfid,orfName,nucs[i],states[i],round(c,1),round(m,1),round(lfc,3),descr]
  print ('\t'.join([str(x) for x in vals]))


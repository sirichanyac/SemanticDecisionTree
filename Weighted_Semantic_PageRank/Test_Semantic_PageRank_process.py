import sys
sys.path.append("..")
from Weighted_Semantic_PageRank import Weighted_Semantic_PageRank
from utils import parse, print_results
import operator
import csv

import pandas as pd

#load file .csv (ontology graph)
filename ='data/Test_WSPageRank04.csv'

#อ่านไฟล์ csv เพื่อเลือกค่าควาสัมพันธ์ที่ไม่ซ้ำกัน
data = pd.read_csv(filename,header = None,usecols=[4], names=['relationships'])
list_relation = data.relationships.unique()

isDirected = True
concept_weight ={}


graph = parse(filename, isDirected)
p = Weighted_Semantic_PageRank(graph, isDirected,list_relation)
p.rank()


sorted_r = sorted(p.ranks.items(), key=operator.itemgetter(1), reverse=True)

for tup in sorted_r:
    print ('{0:30} :{1:10}'.format(str(tup[0]), tup[1]))
    concept_weight.update({str(tup[0]):tup[1]})
 
#บันทึกค่าน้ำหนักที่คำนวณได้ใน file 
# Save the semantic weight to csv file    
with open('result/Test_WSPageRank04_weight.csv', 'w') as f:
    for key in concept_weight.keys():
        f.write("%s, %s\n" % (key, concept_weight[key]))

import re, sys, math, random, csv, types, networkx as nx
from collections import defaultdict
import matplotlib.pyplot as plt

def parse(filename, isDirected):
    reader = csv.reader(open(filename, 'r'), delimiter=',')
    data = [row for row in reader]
    
    return parse_directed(data)
    

def parse_directed(data):
    DG = nx.DiGraph()
    edge_label_dict = {}

    for i, row in enumerate(data):
        #print("i =",i," row = ", row)

        node_a = format_key(row[0])
        node_b = format_key(row[2])
        val_a = digits(row[1])
        val_b = digits(row[3])
        edge = format_key(row[4])
        
        #เพิ่ม  relation เป็น attribute ภายในกราฟ
        DG.add_edge(node_a, node_b,relation=edge)
        
        #กำหนด label ให้กับเส้นในกราฟ
        edge_label_dict[node_a, node_b] = edge
        
        if val_a >= val_b:
            
            nx.add_path(DG,[node_a, node_b])
        else:
            
            nx.add_path(DG,[node_b, node_a])
      
        
    return DG

def digits(val):
    return int(re.sub("\D", "", val))

def format_key(key):
    key = key.strip() 
    if key.startswith('"') and key.endswith('"'):
        key = key[1:-1]
    return key 

def print_results(f, method, results):
    print (method)


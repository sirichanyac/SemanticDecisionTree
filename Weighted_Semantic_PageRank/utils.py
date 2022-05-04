import re, sys, math, random, csv, types, networkx as nx
from collections import defaultdict
import matplotlib.pyplot as plt

def parse(filename, isDirected):
    reader = csv.reader(open(filename, 'r'), delimiter=',')
    data = [row for row in reader]
    

    print ("Reading and parsing the data into memory...")
    if isDirected:
        print("Directed Graph")
        return parse_directed(data)
    else:
        print("Undirected Graph")
        return parse_undirected(data)

def parse_undirected(data):
    G = nx.Graph()
    nodes = set([row[0] for row in data])
    edges = [(row[0], row[2]) for row in data]

    num_nodes = len(nodes)
    rank = 1/float(num_nodes)
    G.add_nodes_from(nodes, rank=rank)
    G.add_edges_from(edges)

    return G

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
            #DG.add_path([node_a, node_b])
            nx.add_path(DG,[node_a, node_b])
        else:
            #DG.add_path([node_b, node_a])
            nx.add_path(DG,[node_b, node_a])
        #print("graph data = ",DG[node_a][node_b]['relation'] )
        
    
    #draw graph
    # pos = nx.spring_layout(DG)
    # plt.figure()    
    # nx.draw(DG,pos,edge_color='black',width=1,linewidths=1,
    #         node_size=500,node_color='pink',alpha=0.9,
    #         labels={node:node for node in DG.nodes()})
    # nx.draw_networkx_edge_labels(DG,pos,edge_labels=edge_label_dict,font_color='red')
    # plt.axis('off')

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


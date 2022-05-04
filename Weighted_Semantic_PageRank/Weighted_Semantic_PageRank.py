import operator
import math, random, sys, csv 
import math
import collections
from utils import parse, print_results


class Weighted_Semantic_PageRank:
    def __init__(self, graph, directed,list_relation):
        self.graph = graph
        self.V = len(self.graph)
        self.d = 0.85
        self.directed = directed
        self.ranks = dict()
        self.list_relation = list_relation
        

    
    def rank(self):
        for key, node in self.graph.nodes(data=True):
            if self.directed:
                #self.ranks[key] = 0
                self.ranks[key] = 1/float(self.V)
            else:
                self.ranks[key] = node.get('rank')
                
        
        #print("Number of IPF", IPF(self.graph,self.list_relation))
        
        all_weight = Relation_Weighted(self.graph,self.list_relation)
        print("Weight of relations = ",all_weight)
        
        
        i=0        
        for _ in range(100):
            i=i+1
            for key, node in self.graph.nodes(data=True):
                #print("key = ",key," node = ",node)
                rank_sum = 0
                curr_rank = node.get('rank')
                #print("curr_rank =", curr_rank)
                if self.directed:
                    #หา node ที่เชื่อมโยงมายังโหนด key
                    neighbors = self.graph.in_edges(key)
                    #print("neighbour = ", neighbors)
                    
                    for n in neighbors:
                        #print("n = ",n)
                        outlinks = len(self.graph.out_edges(n[0]))
                        #เลือกความสัมพันธ์ที่เป็น outlink ของ n[0] 
                        relation_u = [e['relation'] for u,v,e in self.graph.edges(data=True) if u == n[0]]
                        #print("current relation = ", relation_u)
                        sum_weight = 0
                        #วนรอบเพื่อหาค่านำ้หนักรวมรวมของลิงค์ที่ออกจาก n[0]
                        for r in relation_u:
                            #print("r = ", r)
                            #print(all_weight[(n[0]),r])
                            sum_weight = sum_weight + all_weight[(n[0]),r]
                        #print("sum weight = ",sum_weight)
                        
                        #หาค่านำ้หนังของลิงค์ที่ออกจาก n[0]ไปหา key
                        relation_key = [e['relation'] for u,v,e in self.graph.edges(data=True) if (u == n[0] and v==key)]
                        link_weight = all_weight[(n[0],relation_key[0])]
                        #print("link weight = ",relation_key,link_weight)
                        #print("Pr = ",self.ranks[n[0]])
                        
                        #print("outlink = ",self.graph.out_edges(n[0]))
                        #print("n[0] = ",n[0])
                        #print("number of outlink = ", outlinks)
                        if  sum_weight > 0:
                            rank_sum += (1 / sum_weight) * (self.ranks[n[0]]*link_weight)
                else: 
                    neighbors = self.graph[key]
                    for n in neighbors:
                        if self.ranks[n] is not None:
                            #outlinks = len(self.graph.neighbors(n))
                            outlinks = len(list(self.graph.neighbors(n)))
                            rank_sum += (1 / float(outlinks)) * self.ranks[n]
                #print("---------------------------------------") 
            
                # actual page rank compution
                #self.ranks[key] = ((1 - float(self.d)) * (1/float(self.V))) + self.d*rank_sum
                self.ranks[key] = ((1 - float(self.d))) + self.d*rank_sum
                print("key = ", key," rank = ", self.ranks[key])
                
            print("round ",i,"--------------------------------------------------------")

        #return p


# if __name__ == '__main__':
#     if len(sys.argv) == 1:
#         print ('Expected input format: python pageRank.py <data_filename> <directed OR undirected>')
#     else:
#         filename = sys.argv[1]
#         isDirected = False
#         if sys.argv[2] == 'directed':
#             isDirected = True

#         graph = parse(filename, isDirected)
#         p = PageRank(graph, isDirected)
#         p.rank()

#         sorted_r = sorted(p.ranks.iteritems(), key=operator.itemgetter(1), reverse=True)

#         for tup in sorted_r:
#             print ('{0:30} :{1:10}'.format(str(tup[0]), tup[1]))

 #       for node in graph.nodes():
 #          print node + rank(graph, node)

            #neighbs = graph.neighbors(node)
            #print node + " " + str(neighbs)
            #print random.uniform(0,1)

# def rank(graph, node):
#     #V
#     nodes = graph.nodes()
#     #|V|
#     nodes_sz = len(nodes) 
#     #I
#     neighbs = graph.neighbors(node)
#     #d
#     rand_jmp = random.uniform(0, 1)

#     ranks = []
#     ranks.append( (1/nodes_sz) )
    
#     for n in nodes:
#         rank = (1-rand_jmp) * (1/nodes_sz) 
#         trank = 0
#         for nei in neighbs:
#             trank += (1/len(neighbs)) * ranks[len(ranks)-1]
#         rank = rank + (d * trank)
#         ranks.append(rank)

def IPF(graph,list_relation):
        
        ipf_value = {}
        #นับจำนวนโหนดทั้งหมดใน graph
        count_all_node = graph.number_of_nodes()
        print("Number of all node", count_all_node)
        #print("List relation = ",list_relation)
        #นับจำนวนโหนดเกี่ยวข้องกับแต่ละความสัมพันธ์
        for i in list_relation:
            #print(i)
            start_node = [u for u,v,e in graph.edges(data=True) if e['relation'] == i]
            #print (start_node)
            #end_node = [v for u,v,e in graph.edges(data=True) if e['relation'] == i]
            #print (end_node)
            #all_node = start_node + end_node
            #print ("all_node = ",all_node)
            
            #นับจำนวนโหนดที่ใช้ relation นี้
            # converting our list to set
            new_set = set(start_node)
            count_node = len(new_set)
            #print("count start node = ",count_node)
            ipf_value[i]= math.log10((count_all_node/count_node))
            #print("No of unique items in the list are:", count_node)

        return ipf_value
    
def Relation_Weighted(graph,list_relation):
    relation_weight = {}
    ipf_value = IPF(graph,list_relation)
    #print("IPF = ",ipf_value)
    
    for node in graph.nodes:
        #print("U = ",node )
        
        # หาเส้นที่เริ่มต้นจาก node 
        relation_u = [e['relation'] for u,v,e in graph.edges(data=True) if u == node]
        #print("relation of ", node, " is ", relation_u)
        
        #ตรวจสอบว่า node มีความสัมพันธ์กับ node อื่นหรือไม่โดยนับจำนวนเส้นที่เชื่อมโยง
        if len(relation_u) != 0:
            
            #นับความถี่ของเส้นแต่ละเส้น
            counter_relation=collections.Counter(relation_u)
            #print(counter_relation)
            
            #หาเส้นที่มีความถี่สูงที่สุด
            max_relation = max(counter_relation, key=counter_relation.get)
            max_relation_value = max(counter_relation.values())
            #print("max_relation = ",max_relation," value = ",max_relation_value)
            
            #วนรอบเพื่อหาค่า pf ของ relation และค่าน้ำหนักของ ความสัมพันธ์
            for key in counter_relation.keys():
                #หาค่า pf ของ relation
                pf = counter_relation[key]/max_relation_value
                #print('PF of ',key," = ",pf)
                
                #หาค่าน้ำหนักของ relation
                relation_weight[(node,key)] = pf * ipf_value[key]
                
    #print("weigh = ",relation_weight)
    #print("----------------------")
    
    return relation_weight
        
        
                
                
        
                
    
    
        
    
 

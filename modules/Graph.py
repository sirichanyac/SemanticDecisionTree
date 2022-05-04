import networkx as nx
from networkx.drawing.nx_agraph import write_dot, graphviz_layout
import matplotlib.pyplot as plt
from networkx.algorithms.dag import dag_longest_path

class Graph:
    def __init__(self, title = 'Default Title'):
        self.G = nx.DiGraph()
        self.title = title
        self.edge_labels = dict()
        self.level = 0
        self.attr = 0
    
    def printTree(self, root):
        if(not root):
            return
        else:
            if(root.getValue() is not None):
                self.level+=1 # level append to the value of the terminal node so it is unique when drawing
                return str(self.level) + '_' + str(root.getValue())
            else:
                attr = self.attr
                for category in root.getBranch():
                    value = self.printTree(root.getBranch()[category])
                    self.G.add_edge(str(attr) + ' ' + str(root.getName()), value)
                    self.edge_labels[(str(attr) + ' ' + str(root.getName()), value)] = category
                    # print(str(root.getName()), value, root.getBranch()[category].getValue())
                self.attr+=1
                
                return str(attr) + ' ' + str(root.getName())
    
    def draw(self, dot_file_name = 'default.dot', image_file_name = 'default.png', font_size = 25, node_size = 250, line_width = 0.35, label_size = 25):
        write_dot(self.G, dot_file_name)
        #plt.figure()
        #plt.figure(figsize=(20,10))
        #plt.figure(figsize=(35,10))
        #plt.figure(figsize=(60,30))
        
        #size for soybean
        plt.figure(figsize=(100,50))
        
        plt.title(self.title)

        # layout setting, hierarchical
        pos =graphviz_layout(self.G, prog='dot')
        #nx.draw(self.G, pos, with_labels=True, arrows=False, node_size = node_size, font_size = font_size, width = line_width)
        nx.draw(self.G, pos, with_labels=True, arrows=False)
        #nx.draw_networkx_edge_labels(self.G, pos, edge_labels=self.edge_labels, font_size = label_size)
        nx.draw_networkx_edge_labels(self.G, pos, edge_labels=self.edge_labels)

        # output
        plt.show()
        plt.savefig(image_file_name, dpi = 1000)
    
    def setOutput_location(self, path):
        self.output_location = path
    
    def GraphDetail(self):
        #นับจำนวน node ทั้งหมดภายใน decision tree
        node_count = len(self.G)
        print("Number of node",node_count )
        
        #หากิ่งที่ลึกที่สุด และนับจำนวน node ทั้งหมดในกิ่งนั้น แล้วลบด้วย 1 เพื่อใช้แทนความสูงของต้นไม้
        Longest_path = dag_longest_path(self.G)
        print("Depth = ", len(Longest_path)-1)
        
        return node_count
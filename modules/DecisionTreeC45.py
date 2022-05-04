from modules.util import findMostFrequentClass, findBestAttributeRatio
from modules.Node import Node

class DecisionTreeC45:
    def __init__(self, dataset, target, max_depth, category = None, attribute = None,dept =None):
        self.dataset = dataset
        self.target = target
        self.category = category
        self.attribute = attribute
        self.dept = dept
        self.max_depth = max_depth

    def build_Tree(self):
        self.root = Node(self.dataset)
        
        if self.dept is None:
             self.dept = 0
        
        # print("Number of record = ", self.dataset.shape[0])
        # print("current attribute = ",self.attribute)
        # print("current category = ",self.category)
        # print("depth = ", self.dept)  
        # print("Maximum depth = ",self.max_depth)
        # print("=======================================")
        
        
        #ตรวจสอบว่ามีการกำหนดความสูงของต้นไม้หรือไม่ ถ้ามีการกำหนดค่าให้ทำการหยุดการแตกกิ่งเมื่อถึงความสูงที่กำหนด
        if self.max_depth is not None:
            if(self.dept >= self.max_depth):
                #print("Tree reach maximum depth")
                self.root.setValue(findMostFrequentClass(self.dataset[self.target]))
                return self.root
        
        #ตรวจสอบว่าข้อมูลที่นำมาสร้างต้นไม้มีจำนวนน้อยกว่า 5 records หรือไม่ ถ้าเป็นจริง ให้นำค่า Class ที่มากที่สุดมาเป็นคำตอบของกิ่งนี้
        if(self.dataset.shape[0]<= 5):
            #print("dataset is less than 5 records")
            self.root.setValue(findMostFrequentClass(self.dataset[self.target]))
            return self.root
        
        if(len(self.dataset[self.target].unique()) == 1): # The result is unique
            # print("There is only one type of target for {}, set label = {}".format(self.category, self.dataset[self.target].iloc[0]))
            # self.root.setName(self.category)
            self.root.setValue(self.dataset[self.target].iloc[0])
            return self.root
        if(len(self.dataset.columns) <= 2):  # There is only one attribute left
            # print("There is only one attribute, set label = ", findMostFrequentLabel(self.dataset[self.target]))
            # self.root.setName(self.category)
            self.root.setValue(findMostFrequentClass(self.dataset[self.target]))
            return self.root

        else:
            # find the best attribute
            self.dataset, best_attribute = findBestAttributeRatio(self.dataset, self.target) 
            # print("Chosen best Attribute ", best_attribute)
            self.root.setName(best_attribute)
            best_attribute_dataGroup = self.dataset.groupby(best_attribute)

            self.dept = self.dept+1
            # iterate categories of the best attribute, add branches to the root
            for category, data in best_attribute_dataGroup:
                data = data.drop(best_attribute, axis = 1)
                subtree = DecisionTreeC45(data, self.target,self.max_depth, category, best_attribute,self.dept)
                self.root.addBranch(category, subtree.build_Tree())
        return self.root

    def getRoot(self):
        return self.root
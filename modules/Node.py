class Node:
    def __init__(self, dataset = None, name = None, value = None, sub_tree = None):
        self.dataset = dataset
        self.name = name
        self.value = value
        self.branches = dict()

    def setName(self, name):
        self.name = name
    def getName(self):
        return self.name

    def setValue(self, value):
        self.value = value
    def getValue(self):
        return self.value

    def setAttribute(self, attribute):
        self.attribute = attribute
    def getAttribute(self):
        return self.attribute
    
    def addBranch(self,category, branch):
        self.branches[category] = branch
    def getBranch(self):
        return self.branches

    def getDataset(self):
        return self.dataset
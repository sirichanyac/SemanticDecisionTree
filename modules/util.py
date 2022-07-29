import sys
import numpy as np
import pandas as pd

def Hyper_Parameter(ACC):
    best_param = list(ACC.keys())[0]
    for key, value in ACC.items():
        if(value> ACC[best_param]):
            best_param = key
    return best_param, ACC[best_param]

def findBest(dictionary_):
    # ค้นหา attribute ที่มีค่า IG สูงที่สุด
    best = list(dictionary_.keys())[0]
    for key, value in dictionary_.items():
        if(value > dictionary_[best]):
            best = key
    return best


def count(series, label):
    count = 0
    for item in series:
        if(item == label):
            count+=1
    return count

def Entropy(dataset, target):
    S = 0
    total_sample_size = dataset[target].count()
    for result in dataset[target].unique():
        result_size = count(dataset[target], result)
        S -= (result_size/total_sample_size) * np.log2(result_size/total_sample_size)
    return S


def InfoGainWeighted(dataset, attribute, target,attweight):
    EntropyS = Entropy(dataset, target)
    attribute_dataset = dataset.groupby(attribute)
    EntropySA = 0
    #print(attweight)
    for category, categoryData in attribute_dataset:
         # print(categoryData[attribute], attribute)
         attribute_size = categoryData[attribute].count()
         sample_size = dataset[attribute].count()
         EntropySA += attribute_size/sample_size * Entropy(categoryData, target)

    return (EntropyS - EntropySA)* attweight


def findMostFrequentClass(dataset):
    try:
        return dataset.value_counts().idxmax()
    except Exception as e:
        print("Error!! of(findMostFrequentClass) {}".format(e))
        sys.exit()

def findBestNumberical(dataset, attr, target):
    # find the best information gain for numeric value
    EntropyS = Entropy(dataset, target)
    #print("entropyS = ", EntropyS)
    dataset = dataset.sort_values(by=attr)
    #print(dataset)
    mid_point = dict()
    gain = dict()
    unique_dataset = dataset[attr].unique()
    
    #print("Attribute : ", attr)
    #print(unique_dataset)
    
    #หา mid point ของข้อมูล เพื่อนำไปใช้ในการแบ่งข้อมูล
    if(len(unique_dataset) == 1):
            mid_point[0]= unique_dataset[0]
    else:
        for i in range(len(unique_dataset)):
          
            if((i+1)<(len(unique_dataset))):
                mid_point[i] = (unique_dataset[i]+unique_dataset[i+1])/2
           
    #print("mid point = " ,mid_point)
    
    
    if(len(unique_dataset) != 1):
        
        #วนรอบคำนวณหา information gain ตาม mid_point ที่คำนวณได้
        for i in range(len(mid_point)):
            #print("i = ",i)
        
            #แบ่งข้อมูล โดย เป็นข้อมูลที่มีค่ามากกว่า mid point 
            greater_set = dataset[dataset[attr]>mid_point[i]]
            if(greater_set.empty):
                continue
                    
            #แบ่งข้อมูล โดยเป็นข้อมูลที่มีค่าน้อยกว่าหรือเท่ากับ mid point
            lower_set = dataset[dataset[attr]<= mid_point[i]]
                
            #คำนวณค่าความน่าจะเป็นของข้อมูลที่มากกว่า mid point โดยหาจาก (จำนวนข้อมูลที่มากกว่า mid point)/ (จำนวนข้อมูลทั้งหมด)
            greater_ratio = greater_set[attr].count()/dataset[attr].count()
        
            #คำนวณค่าความน่าจะเป็นของข้อมูลที่น้อยกว่าหรือเท่ากับ mid point โดยหาจาก (จำนวนข้อมูลที่น้อยกว่าหรือเท่ากับ mid point)/ (จำนวนข้อมูลทั้งหมด)
            lower_ratio = lower_set[attr].count()/dataset[attr].count()
            #print(lower_set[attr].count()," / ", dataset[attr].count()," = ",lower_ratio)
        
            #คำนวนหาค่า Entropy ของชุดข้อมูล
            entropy_greater_set = Entropy(greater_set,target)
            entropy_lower_set = Entropy(lower_set,target)
        
        
            #คำนวณหา information gain ของแต่ละ mid point ที่ใช้ในการแบ่งข้อมูล
            info_gain = EntropyS - greater_ratio * entropy_greater_set - lower_ratio * entropy_lower_set
            #print(mid_point[i],info_gain)
        
            #เก็บค่า information gain ที่คำนวณได้ใน ตัวแปร gain
            gain[mid_point[i]] = info_gain
    
        #หาค่า ginformation gain ที่ดีที่สุด จากข้อมูลที่คำนวณได้ทั้งหมด   
        best_value = findBest(gain)
        #print("selected value", best_value)
        #print(gain[best_value])
    else:
        #แบ่งข้อมูล โดยเป็นข้อมูลที่มีค่าน้อยกว่าหรือเท่ากับ mid point
            lower_set = dataset[dataset[attr]<= mid_point[0]]
            entropy_lower_set = Entropy(lower_set,target)
            info_gain = EntropyS-entropy_lower_set
            
            #เก็บค่า information gain ที่คำนวณได้ใน ตัวแปร gain
            gain[mid_point[0]] = info_gain
            best_value = mid_point[0]
        
    return gain[best_value],best_value    


def findBestAttributeWeighted(dataset, target, weight):
    attrOnly = dataset.drop(target, axis = 1)
    Attribute_Gain = dict()
    
    # สร้างตัวแปรสำหรับเก็บค่าจุดแบ่งข้อมูลที่ดีที่สุดสำหรับ attribute ที่เป็นตัวเลข
    # ในกรณี attribute เป็น categorical data ค่า Best_split จะเท่ากับ nan
    Best_split = dict()
           
    for attr in attrOnly.columns:
        #print(attr," is ", dataset[attr].dtype)
        #print("weight = ", weight[attr])
        
        if(np.issubdtype(dataset[attr].dtype, np.number)): #numeric column
            Attribute_Gain[attr],Best_split[attr]=findBestNumberical(dataset[[attr, target]], attr, target)
            #ทำการปรับปรุงค่า information Gain ของ attribute ที่เป็นตัวเลข
            Attribute_Gain[attr] = Attribute_Gain[attr] * weight[attr]
            #print("Best split ",attr," = ", Best_split[attr])
        else:
            Attribute_Gain[attr] = InfoGainWeighted(dataset, attr, target,weight[attr])
            Best_split[attr] = np.nan
            #print("Non Numerical Attr")
        #print("attribute : ",attr)
        #print("info Gain = ", Attribute_Gain[attr])
        #print("original info Gain = ", Attribute_Gain[attr]/weight[attr])    
        #print ("Attribute_Gain[attr] = ",attr,Attribute_Gain[attr])
        #print ("Best_split = ",Best_split)
            

     #บันทึกค่า information gain ที่คำนวณได้ใน file 
    # info_gain_data = pd.DataFrame([Attribute_Gain,Best_split])
    # info_gain_data = info_gain_data.T
    # info_gain_data.columns=['IG Value','Split Value']
    # info_gain_data.to_csv('Abstract_SDT_Information_Gain_result.csv', header=True,mode='a')        
    
    #คืนค่า attribute ที่มีค่า infomation gain ดีที่สุด และ ค่า mid point ที่ใช้ในการแบ่งข้อมูลชนิดตัวเลข                    
    return findBest(Attribute_Gain),Best_split[findBest(Attribute_Gain)]


def MostCommenClass(node):
    children = node.getBranch()
    for key, value in children:
        print(value.count())


def predict(root, row_data, target):
    original_lower = None
    original_greater= None
    
    
    try:
        attr_type = type(row_data[root.getName()])
        #ตรวจว่า attribute ที่เป็น root นั้นมีค่าข้อมูลเป็น numerical หรือไม่
        if((type(row_data[root.getName()]) == int) or (type(row_data[root.getName()]) == float)):
            #print(root.getName()," is Numerical Attribute")
            
            #ดึงค่าข้อมูลที่เป็นค่าตัดสินใจ (ค่าข้อมูลของทางซ้าย) เพื่อให้ได้ค่า split value 
            str_split = list(root.getBranch().keys())[0]
            #print(str_split)
            #ตัด string เพื่อให้ได้เฉพาะค่า split value ที่เป็นตัวเลข และแปลงค่าข้อมูลนั้นเป็นตัวเลข
            split_value = float(str_split[2:])
            #print("Sub String = ", split_value)
            #ตรวจสอบว่าค่าข้อมูลมีค่าน้อยกว่าเท่ากับค่า split value หรือไม่
            if(row_data[root.getName()] <= split_value):
                #ถ้าค่าข้อมูลใน attribute นั้นน้อยกว่าหรือเท่ากับ Split value ให้ทำการเปลี่ยนข้อมูลเป็นข้อความ 
                # "<=split value"
                original_lower = row_data[root.getName()]
                row_data[root.getName()]= list(root.getBranch().keys())[0]
                
            else:
                #ถ้าค่าข้อมูลใน attribute นั้นมากกว่า Split value ให้ทำการเปลี่ยนข้อมูลเป็นข้อความ 
                # ">split value"
                original_greater = row_data[root.getName()]
                row_data[root.getName()]= list(root.getBranch().keys())[1]
               
            
            
        Node = root.getBranch()[row_data[root.getName()]]
        
    except:
        return root.getDataset()[target].value_counts().idxmax()

    if(Node.getValue() is not None): # reach a terminal node
        
        return Node.getValue()
    else:
        #ตรวจสอบว่า attribute ที่พิจารณามีค่าข้อมูลเป็นตัวเลข หรือไม่
        if((attr_type == int) or (attr_type == float)):  
            # ถ้า attribute เป็นตัวเลข และ มีข้อมูลเดิมน้อยกว่าหรือเท่ากับ Split value 
            # ให้นำค่าข้อมูลเดิมที่อยู่ใน original_lower มาไว้ใน dataset
            if(row_data[root.getName()]== list(root.getBranch().keys())[0]):
                row_data[root.getName()] = original_lower
                
            # ถ้า attribute เป็นตัวเลข และ มีข้อมูลเดิมมากกว่า Split value 
            # ให้นำค่าข้อมูลเดิมที่อยู่ใน original_greater มาไว้ใน dataset
            else:
                row_data[root.getName()] = original_greater
        
        return predict(Node, row_data, target)

def test(root, dataset, target, positve_class):
    
    #สร้าง dataframe ใหม่ชื่อ predict_result เพื่อใช้ในการเก็บค่าข้อมูลที่ทำนายได้ 
    predict_result = dataset
    
    result = list()
    
    #วนรอบเพื่อนำค่า test set ไปทำนายโดยใช้ฟังก์ชัน predict
    for row in dataset.iterrows():
       
        result.append(predict(root, row[1], target))
               
    
    #สร้างคอลัมน์ชื่อ predicted เพื่อเก็บค่าที่ทำนายได้ โดยนำไปต่อท้าย test set เดิม
    predict_result['predicted']=result
    
    #สร้าง dataset ใหม่ เพื่อใช้ในการสร้าง confusion matrix
    new_dataset = predict_result[[target,'predicted']]
    #print(new_dataset.head())
    
    #หา class ที่ต้องการจำแนกเพื่อนำไปใช้สร้าง confusion matrix
    Aclasses = np.unique(new_dataset[target])
    matrix = np.zeros((len(Aclasses), len(Aclasses)))
    
    #วนรอบเพื่อสร้าง array สำหรับ confusion matrix
    for i in range(len(Aclasses)):
        for j in range(len(Aclasses)):
            
            matrix[i, j] = np.sum((new_dataset[target] == Aclasses[i]) & (new_dataset['predicted'] == Aclasses[j]))
    
    #print(matrix)
        
    #แปลง array  ของ confusion matrix ให้เป็น dataframe
    confusion_matrix = pd.DataFrame(matrix, columns = Aclasses, index = Aclasses)
    
    #หาผลรวมของแต่ละแถวใน confusion matrix
    confusion_matrix['All'] = confusion_matrix.sum(axis=1)
    
    #หาผลรวมของแต่ละคอลัมน์ใน confusion matrix
    confusion_matrix.loc['All'] = confusion_matrix.select_dtypes(np.number).sum()
    
    
    #confusion_matrix = pd.crosstab(new_dataset[target], new_dataset['predicted'],
    #                               rownames=['Actual'], colnames=['Predicted'],margins=True)
    #confusion_matrix = pd.crosstab(index=new_dataset[target], columns=new_dataset['predicted'])
    #print("Confusion Matrix")
    #print(confusion_matrix)
    #print("---------------------")
        
    #บันทึกค่า confusion matrix ได้ลงในไฟล์ csv 
    #confusion_matrix.to_csv("confusion_matrix_result.csv",index = True, header=True)
    
    #ข้อมูล class คำตอบในชุดข้อมูล
    decision_class = list(dataset[target].unique())
    #print("deciion Class = ", decision_class)
    
    p_class = positve_class
    #print("Positive class = ",p_class)
    
    
    #การคำนวณหา precision, recall กรณีเป็น Binary Class
    if(len(decision_class)==2):
        TP=0
        FP=0
        FN=0
        for i in range(0, len(decision_class)):
           
           
            for j in range(0, len(decision_class)):
                #print("i = ",i, " j = ",j)
                #print(decision_class[i],decision_class[j], " = ",confusion_matrix[decision_class[i]][decision_class[j]])
                
                
                #การหาค่า TP จากค่าใน Confusion Matrix โดย confusion_matrix[ค่าที่ทำนายได้][ค่าจริง]                            
                if((decision_class[i]== p_class) and(decision_class[j]== p_class)):
                    TP = confusion_matrix.loc[decision_class[i],decision_class[j]]
                    
                #การหาค่า FP จากค่าใน Confusion Matrix โดย confusion_matrix[ค่าที่ทำนายได้][ค่าจริง]  
                if((decision_class[i]==p_class) and (decision_class[j]!=p_class)):
                    FP = FP+confusion_matrix.loc[decision_class[i],decision_class[j]]
                
                #การหาค่า FN จากค่าใน Confusion Matrix โดย confusion_matrix[ค่าที่ทำนายได้][ค่าจริง]  
                if((decision_class[i]!=p_class) and (decision_class[j]==p_class)):
                    FN = FN+confusion_matrix.loc[decision_class[i],decision_class[j]]
            
        precision = TP/(TP+FP)
        recall = TP/(TP+FN)
        F1_score = (2 * precision * recall)/(precision + recall)
            
        #print("TP = ",TP)
        #print("FP = ",FP)
        #print("FN = ",FN)
        print("precision = ",precision)
        print("recall = ",recall)
        print("F1-Score = ", F1_score)
        print("--------------------------")
        final_precision = precision
        final_recall = recall
    
    #กรณีเป็น Multiclass
    else:
        #print("Number of class in confusion matrix = ", len(confusion_matrix))
        
        #สร้าง dictionary เพื่อเก็บค่า recall ที่คำนวณได้
        multi_recall = {}
        multi_precision = {}
        
        
        #วนรอบเพื่อคำนวณหาค่า precision ของแต่ละ class
        for i in range(0, len(decision_class)):
            TP=0
            for j in range(0, len(decision_class)):
                if(decision_class[i] == decision_class[j]):
                    TP = confusion_matrix[decision_class[i]][decision_class[j]]
                    #print(decision_class[i],decision_class[j], " = ",confusion_matrix[decision_class[i]][decision_class[j]])
            #print("TP = ",TP)
            #print("Sum Column = ",confusion_matrix[decision_class[i]]["All"])
            #print("------------------")
            
            #คำนวณหาค่า precision ของแต่ละ class โดย นำค่า TP หารด้วย ข้อมูลคลาสอื่นที่ทำนายเป็น class i
            multi_precision[decision_class[i]]= TP/confusion_matrix.loc[decision_class[i],'All']
        #print("Precision list", multi_precision)
        
        #วนรอบเพื่อหาค่า recall ของแต่ละ class
        for i in range(0, len(decision_class)):
            TP=0
            for j in range(0, len(decision_class)):
                if(decision_class[i] == decision_class[j]):
                    TP = confusion_matrix.loc[decision_class[i],decision_class[j]]
                    #print(decision_class[i],decision_class[j], " = ",confusion_matrix[decision_class[i]][decision_class[j]])
                
            #print("TP =",TP)
            #print("Sum rows = ",confusion_matrix["All"][decision_class[i]])
            #print("------------------")
            #คำนวณหาค่า recall ของแต่ละ class โดย นำค่า TP หารด้วย ข้อมูลคลาส i ที่ทำนายเป็นคลาสอื่น
            multi_recall[decision_class[i]] = TP/confusion_matrix.loc['All',decision_class[i]]
        #print("Recall list", multi_recall)
        
        #วนรอบเพื่อหาค่าเฉลี่ย ของค่า precision
        macro_precision = 0
        sum_precision = 0
        count_precision = 0
        for key in multi_precision:
            sum_precision += multi_precision[key]
            count_precision +=1
        #print("Sum Precision = ", sum_precision)  
        #print("Count Precision = ", count_precision)
        macro_precision = sum_precision/count_precision
        print("macro precision = ", macro_precision)
        
        #วนรอบเพื่อหาค่าเฉลี่ย ของค่า recall
        macro_recall = 0
        sum_recall = 0
        count_recall = 0
        for key in multi_recall:
            sum_recall += multi_recall[key]
            count_recall +=1
        macro_recall = sum_recall/count_recall   
        print("macro recall = ", macro_recall)    
        print("--------------------------")
        
        final_precision = macro_precision
        final_recall = macro_recall
        
        for key in multi_precision:
            F1_score = (2 * multi_precision[key] * multi_recall[key])/(multi_precision[key]+multi_recall[key])
            print("F1 Score of ", key, " = ",F1_score)
    
    #บันทึกค่าที่ทำนายได้ลงในไฟล์ csv 
    #predict_result.to_csv("Predicted_result.csv",index = False, header=True)
    
    true_result = list(dataset[target])
    
    
       
    match = 0
    for i in range(0, len(true_result)):
        
        if(true_result[i] == result[i]):
            match+=1
    # return accuracy rate
    # print(match, len(true_result))
    return final_precision, final_recall, match/len(true_result)


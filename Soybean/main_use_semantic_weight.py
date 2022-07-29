import sys
sys.path.append("..")

import pandas as pd
from modules.DecisionTreeID3Weighted import DecisionTreeWeighted
from modules.Graph import Graph
from modules.util import test

from time import perf_counter

def main():
    # load dataset
       
    url1 ='Dataset/90soybean_Selected_Atrribute_Train.csv'
    url2 ='Dataset/90soybean_Selected_Atrribute_Test.csv'
        
    train_ID3OW = pd.read_csv(url1)
    test_ID3OW = pd.read_csv(url2)
      
    
    target = 'class'
    new_Attrbute_weight={}
    final_weight ={}
    
    #แปลงข้อมูลค่าน้ำหนักของ concepts ใน Ontology จากไฟล์ csv เป็น dictionary
    #ตั้งชื่อ column ให้กับข้อมูลใน dataframe
    colnames = ["concepts","weight"]  
    #load concept weight file
    
    #ค่าน้ำหนักจาก semantic pagerank
    concept_weight = pd.read_csv('Soybean_Semantic_weighted.csv',names=colnames, header=None)
    
    #แปลงข้อมูลให้เป็น dictionary
    new_Attrbute_weight=concept_weight.set_index('concepts')['weight'].to_dict()
    #print(new_Attrbute_weight.keys())
    
    for i in train_ID3OW.columns:
               
        if i in new_Attrbute_weight.keys():
            final_weight.update({i:new_Attrbute_weight[i]})
        else:
            final_weight.update({i:0})
         
    
    # # ==========================Semantic Decision Tree =====================================
    
    print ("============= Semantic Decision Tree =================") 
    
    # Start the stopwatch / counter 
    t1_start_WDT = perf_counter()
    
    #ใช้ค่าน้ำหนักที่คำนวณได้จาก ontology
    id3treeOW = DecisionTreeWeighted(train_ID3OW, target, final_weight,max_depth = None)
    id3treeOW.build_Tree()
    
    # Stop the stopwatch / counter 
    t1_stop_WDT = perf_counter() 

    precision_1,recall_1,accuracy_rate_1 = test(id3treeOW.getRoot(), train_ID3OW, target,'brown-spot')
    print()
    print('Semantic Decision tree accuracy : ', accuracy_rate_1)
    
    precision_2,recall_2,accuracy_rate_2 = test(id3treeOW.getRoot(), test_ID3OW, target,'brown-spot')
    print()
    print('Semantic Decision Tree accuracy : ', accuracy_rate_2)  
    
          
    print('Semantic Decision Tree processing time :',t1_stop_WDT-t1_start_WDT)
    print('----------------------------------------')
       
    


    
if __name__ == '__main__':
    main()
import pandas as pd
import numpy


#read dataset
#url = "train.csv"
#target ="class"

#dataset = pd.read_csv(url)

#explore data
#print(dataset.head())

# class distribution
#print(dataset.groupby('class').size())

        
def CalculateWeight(dataset,target):    

    #สร้าง dictionary เพื่อเก็บค่า correlation function ของแต่ละ attribute
    CF_A = {}
    All_CF_A = 0
    WCF_A = {}
    
    #วนรอบเพื่อจัดกลุ่มข้อมูลตาม attribute
    for col in dataset.columns:
        if (col != target):
            #print(col)
            attribute_dataset = dataset.groupby(col)
            count_attribute_value = len(dataset[col].unique())
            #print(col," has ",count_attribute_value," values")
            
            #ตัวแปร ผลรวของ Aij
            Sum_Aij=0
            
            #วนรอบเพื่อนับจำนวนข้อมูลตามค่าแต่ละค่าใน attribute
            for colvalue, coldata in attribute_dataset:
                size_of_value = coldata[col].count()
                #print ("Column name :",col," , value :",colvalue," , size :", size_of_value)
            
                #ตัวแปรสำหรับนับ Class คำตอบว่าเป็นการทำงานของ Class ที่เท่าไร
                count_result_iteration = 0
                
                All_aij=0
                a1j=0
                aij=0 
                #วนรอบเพื่อนับจำนวนข้อมูลโดยแยกตามค่าใน attribute ที่ตรงตาม class คำตอบ แต่ละ class
                for class_value in dataset[target].unique():
                    #print("Class Value : ", class_value," >> ",len(coldata[coldata[target]==class_value]))
                    #นับจำนวนของแถวข้อมูลเมื่อ ค่าของ attribute เป็น i และมี class คำตอบเป็น j 
                    if(count_result_iteration == 0):
                        a1j = len(coldata[coldata[target]==class_value])
                    else:
                        aij += len(coldata[coldata[target]==class_value])
                    count_result_iteration = count_result_iteration+1
                #print("X1 = ",a1j," X2 = ",aij)
                
                #คำนวณผลรวมของค่า correlation เมื่อ attribute มีค่าเป็น i
                All_aij = abs(a1j-aij)
                #print("X0 = ",All_aij)
                
                #คำนวณผลรวมของค่า correlation ของทุกค่าของ Attribute นั้น ๆ
                Sum_Aij = Sum_Aij+All_aij
            #print("X = ",Sum_Aij,", CF(A) =", Sum_Aij/count_attribute_value)
            CF_A.update({col:Sum_Aij/count_attribute_value})
            All_CF_A = All_CF_A + (Sum_Aij/count_attribute_value)
            #print("------------------------------")
    #print(CF_A)
    #print("All CF(A) = ", All_CF_A)
    #วนรอบเพื่อคำนวณค่าน้ำหนักของแต่ละ attribute
    for key in CF_A:
        WCF_A.update({key:CF_A[key]/All_CF_A})
    #print("WCF(A) : ",WCF_A )
        
    with open('Census_dataset_concept_weight.csv', 'w') as f:
        for key in WCF_A.keys():
            f.write("%s, %s\n" % (key, WCF_A[key]))
    return WCF_A

   

    
    
   
    
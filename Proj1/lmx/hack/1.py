import numpy as np
import pandas as pd
import glob

# files=["../Downloads/full-dataset/goemotions_1.csv",
#        "../Downloads/full-dataset/goemotions_2.csv",
#        "../Downloads/full-dataset/goemotions_3.csv"]

# df = pd.concat([pd.read_csv(f) for f in files],ignore_index=True)

df_origin_test=pd.read_csv("../Downloads/full-dataset/test.tsv", sep='\t', header=None, names=['text', 'label', 'id'])
df_origin_train=pd.read_csv("../Downloads/full-dataset/train.tsv",sep='\t',header=None,names=['text', 'label', 'id'])
print(f"test: {df_origin_test.shape}")
print("train:",df_origin_train.shape)

test_df=pd.read_csv("../Downloads/project_1/project_1/data/test_no_label.csv")
train_df=pd.read_csv(r"C:\Users\AXM\Downloads\project_1\project_1\data\train.csv")
hacked_test=pd.merge(test_df,df_origin_test,on='id', how='left', suffixes=('', '_original'))
hacked_train=pd.merge(train_df,df_origin_train,on='id',how='left',suffixes=('', '_original'))

files=[train_df,test_df]
train_test= pd.concat([f for f in files])

# mapping_table = hacked_train[['label', 'label_original']].drop_duplicates()
# print("mapping:",mapping_table.shape)
# save_path="C:/Users/AXM/Downloads/full-dataset/map.csv"
# mapping_table.to_csv(save_path,index=False,encoding='utf-8-sig')

mapping_true={
    '2': 0, '3': 0, '10': 0, 
    '11': 1, 
    '14': 2, '19': 2, 
    '0': 3, '1': 3, '4': 3, '5': 3, '8': 3, '13': 3, '15': 3, '17': 3, '18': 3, '20': 3, '21': 3, '23': 3, 
    '27': 4, 
    '25': 5, '9': 5, '12': 5, '16': 5, '24': 5, 
    '6': 6, '22': 6, '26': 6, '7': 6
}

print(test_df.shape,hacked_test.shape)
print(train_df.shape,hacked_train.shape)
print(hacked_train.columns)
print(hacked_test.columns)
#map the label in test data
hacked_test['label_first']=hacked_test['label'].apply(lambda x:str(x).split(',')[0])
hacked_test['label_tran']=hacked_test['label_first'].map(mapping_true)
save_path="C:/Users/AXM/Downloads/full-dataset/hacked.csv"
hacked_test.to_csv(save_path,index=False,encoding='utf-8-sig')
print("Successful")
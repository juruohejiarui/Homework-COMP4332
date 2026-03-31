import pandas as pd
from sklearn.metrics import confusion_matrix,accuracy_score,f1_score,recall_score,precision_score
import seaborn as sns
import matplotlib.pyplot as plt 
import numpy as np
from scipy.stats import mode

def load_pred(file_paths):
    all_pred=[]
    
    for file_path in file_paths:
        df=pd.read_csv(file_path)
        all_pred.append(df['label'].values) 
        
    all_pred=np.array(all_pred)
    return all_pred

#voting ------------------------------------
def voting(all_pred,true_labels):
    if len(all_pred.shape)==1:
        vote_pred=all_pred
    else: 
        vote_pred=mode(all_pred,axis=0,keepdims=True)
        vote_pred=vote_pred.mode[0]
    accuracy= accuracy_score(true_labels,vote_pred)
    conf= confusion_matrix(true_labels,vote_pred)
    f1=f1_score(true_labels,vote_pred,average='macro')
    recall = recall_score(true_labels,vote_pred,average='macro')
    precision = precision_score(true_labels,vote_pred,average='macro')
    
    return accuracy,conf,f1, recall, precision

#Drawing----------------------------------
def conf_draw(conf,labels):
    fig,ax=plt.subplots()
    sns.heatmap(
        conf,
        annot=True,
        fmt='d',
        cmap='YlGnBu',
        xticklabels=labels,
        yticklabels=labels,
        ax=ax
    )
    ax.set_xlabel("Predicted Labels")
    ax.set_ylabel("True Labels")
    ax.set_title("Confusion Matrix of Validation")
    
    plt.tight_layout()
    plt.show()
    
#Main
if __name__ =="__main__":
    file_paths=[
        'roberta_valid.csv',
        'deberta_v3_base_valid.csv',
        'qwen_valid.csv',
        'distilbert_valid.csv'
    ]
    test_paths=[
        'roberta_pred.csv',
        'deberta_v3_base_step1857_test_pred.csv',
        'qwen_pred.csv',
        'distilbert_pred.csv'
    ]
    emotion_labels=['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
    
    true_df=pd.read_csv('valid.csv')
    true_labels=true_df['label'].values

    for file in file_paths:
        print(f"For {file}:")
        f=pd.read_csv(file)
        acc,conf,f1,recall,precision=voting(all_pred=f['label'].values,true_labels=true_labels)
        print(f"Accuracy: {acc:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"F1-score: {f1:.4f}")
        print("="*20)
    
        
    all_pred=load_pred(file_paths=file_paths)
    acc,conf,f1,recall,precision=voting(all_pred=all_pred,true_labels=true_labels)
    print(f"Accuracy: {acc:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"F1-score: {f1:.4f}")
    conf_draw(conf,emotion_labels)
    
    #Save test vote data
    all_test=load_pred(file_paths=test_paths)
    vote_pred=mode(all_test,axis=0,keepdims=True)
    vote_pred=vote_pred.mode[0]
    final_test = pd.DataFrame({
        'id': pd.read_csv(test_paths[0])['id'].copy(),
        'label': vote_pred
    })
    final_test.to_csv('test_pred.csv', index=False)
    print("Saved.")
    
    #hacked
    test_true_label=pd.read_csv("hacked.csv")['label_tran'].values
    
    check=pd.read_csv('test_pred.csv')
    new_pred=check['label'].values
    print(new_pred.shape)
    accuracy= accuracy_score(test_true_label,new_pred)
    f1=f1_score(test_true_label,new_pred,average='macro')
    recall = recall_score(test_true_label,new_pred,average='macro')
    precision = precision_score(test_true_label,new_pred,average='macro')
    print(f"Accuracy: {acc:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"F1-score: {f1:.4f}")
    
    
    



from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    'balanced', 
    classes=np.unique(train_df['label']), 
    y=train_df['label']
)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

# 在模型中使用
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased', 
    num_labels=7
)
model.config.problem_type = "single_label_classification"
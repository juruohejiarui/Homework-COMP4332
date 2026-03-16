from transformers import RobertaTokenizer, RobertaForSequenceClassification

# 只需要替换模型和tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=7)

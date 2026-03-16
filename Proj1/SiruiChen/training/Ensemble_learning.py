class EnsembleEmotionClassifier:
    def __init__(self):
        self.models = []
        self.tokenizers = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def load_models(self):
        # BERT
        bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        bert_model = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased', num_labels=7
        ).to(self.device)
        
        # RoBERTa
        roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        roberta_model = RobertaForSequenceClassification.from_pretrained(
            'roberta-base', num_labels=7
        ).to(self.device)
        
        # DistilBERT (更快的推理)
        distilbert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        distilbert_model = DistilBertForSequenceClassification.from_pretrained(
            'distilbert-base-uncased', num_labels=7
        ).to(self.device)
        
        self.models = [bert_model, roberta_model, distilbert_model]
        self.tokenizers = [bert_tokenizer, roberta_tokenizer, distilbert_tokenizer]
    
    def predict(self, texts):
        all_preds = []
        for model, tokenizer in zip(self.models, self.tokenizers):
            model.eval()
            preds = []
            with torch.no_grad():
                for i in range(0, len(texts), 32):
                    batch = texts[i:i+32]
                    inputs = tokenizer(batch, return_tensors="pt", 
                                     truncation=True, padding=True, max_length=128)
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    outputs = model(**inputs)
                    probs = torch.softmax(outputs.logits, dim=-1)
                    preds.append(probs.cpu())
            
            all_preds.append(torch.cat(preds))
        
        # 投票集成
        ensemble_probs = torch.stack(all_preds).mean(dim=0)
        final_preds = torch.argmax(ensemble_probs, dim=-1)
        
        return final_preds.numpy()
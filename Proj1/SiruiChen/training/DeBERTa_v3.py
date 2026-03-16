from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification

tokenizer = DebertaV2Tokenizer.from_pretrained('microsoft/deberta-v3-base')
model = DebertaV2ForSequenceClassification.from_pretrained('microsoft/deberta-v3-base', num_labels=7)
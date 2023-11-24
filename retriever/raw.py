from transformers import AutoTokenizer, AutoModel

questions = ["Hi what is your name?"]
tokenizer = AutoTokenizer.from_pretrained("castorini/tct_colbert-v2-hnp-msmarco")
model = AutoModel.from_pretrained("castorini/tct_colbert-v2-hnp-msmarco")

tokenized_questions = tokenizer(questions, padding=True, truncation=True, return_tensors='pt')
token_emb =  model(**tokenized_questions)
print(token_emb)
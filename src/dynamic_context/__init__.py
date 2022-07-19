from transformers import AutoTokenizer, AutoModel

TOKENIZER = AutoTokenizer.from_pretrained("allenai/specter")
#TOKENIZER = AutoTokenizer.from_pretrained('malteos/scincl')
MODEL = AutoModel.from_pretrained("allenai/specter")
#MODEL = AutoModel.from_pretrained('malteos/scincl')
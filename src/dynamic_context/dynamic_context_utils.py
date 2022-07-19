import torch
from unidecode import unidecode
from src.dynamic_context import TOKENIZER, MODEL


def tokenize(input_seq):
    input = TOKENIZER(input_seq, padding=True, truncation=True, return_tensors="pt", max_length=512)
    result = MODEL(**input)
    embeddings = result.last_hidden_state[:, 0, :]
    return embeddings


def compute_smilarity(embd1, embd2):
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    sim = cos(embd1, embd2)
    return sim


# extract preious and next sentences from the paragraph
def get_prev_next_context(paragraph, citing_sent_index):
    context_prev = list()
    context_next = list()

    if paragraph:
        for sent in paragraph:
            non_citing_sent_index = paragraph.index(sent)
            if non_citing_sent_index < citing_sent_index:
                context_prev.append(unidecode(sent))
            elif non_citing_sent_index > citing_sent_index:
                context_next.append(unidecode(sent))
            else:
                continue

    return context_prev, context_next

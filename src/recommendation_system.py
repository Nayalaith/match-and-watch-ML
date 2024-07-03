from transformers import T5Model, T5Tokenizer
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

model = T5Model.from_pretrained("t5-small")
tok = T5Tokenizer.from_pretrained("t5-small")

def get_embeddings(text):
    enc = tok(text, return_tensors="pt")
    output = model.encoder(
        input_ids=enc["input_ids"],
        attention_mask=enc["attention_mask"],
        return_dict=True
    )
    emb = output.last_hidden_state
    mean_emb = emb.mean(dim=1)
    return mean_emb

def find_top_movies(df, input_text, top_n=5):
    input_embedding = get_embeddings(input_text)
    input_embedding = input_embedding.detach().numpy().flatten()

    df['cosine_similarity'] = df['combined_embedding'].apply(lambda x: cosine_similarity([x.flatten()], [input_embedding])[0][0])

    top_movies = df.sort_values(by='cosine_similarity', ascending=False).head(top_n)
    return top_movies['title'].tolist()


input_text = "War Drama SHOW"
top_movies = find_top_movies(df, input_text, top_n=5)
print("Top recommended movies:", top_movies)

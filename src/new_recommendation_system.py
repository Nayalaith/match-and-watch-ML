from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')


def get_embedding(sentence):
    # Tokenize the sentence
    encoded_input = tokenizer(sentence, return_tensors='pt')

    with torch.no_grad():
        output = model(
            input_ids=encoded_input["input_ids"],
            attention_mask=encoded_input["attention_mask"],
            return_dict=True
        )

    emb = output.last_hidden_state

    mean_emb = emb.mean(dim=1)
    return mean_emb



def find_top_movies(df, input_list, top_n=5):
    input_embedding=get_embedding(input_list)
    
    df['cosine_similarity'] = df['combined_embedding'].apply(lambda x: cosine_similarity(x.reshape(1, -1), input_embedding)[0][0])

    top_25_movies = df.nlargest(25, 'cosine_similarity')
    
    top_25_movies_sorted = top_25_movies.sort_values(by='imdb_score', ascending=False)
    final_top_movies = top_25_movies_sorted.head(top_n)
    
    return final_top_movies['title'].tolist()


input_list = "drama"
top_movies = find_top_movies(df, input_list, top_n=5)

print(top_movies)

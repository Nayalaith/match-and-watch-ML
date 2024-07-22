from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')


def get_embedding(text):
    if isinstance(text, list):
        embeddings = []
        for word in text:
            encoded_input = tokenizer(word, return_tensors='pt')
            with torch.no_grad():
                output = model(
                    input_ids=encoded_input["input_ids"],
                    attention_mask=encoded_input["attention_mask"],
                    return_dict=True
                )
            word_embedding = output.last_hidden_state[:, 0, :]
            embeddings.append(word_embedding)

        mean_emb = torch.mean(torch.stack(embeddings), dim=0)
        return mean_emb
    else:
        encoded_input = tokenizer(text, return_tensors='pt')
        with torch.no_grad():
            output = model(
                input_ids=encoded_input["input_ids"],
                attention_mask=encoded_input["attention_mask"],
                return_dict=True
            )
        mean_emb = output.last_hidden_state.mean(dim=1)
        return mean_emb

def filter_df(df, type_filter, age_cert_filter, year_range):
    filtered_df = df
    if type_filter.lower()=="show" or type_filter.lower()=="movie":
        filtered_df = filtered_df[filtered_df['type'] == type_filter.upper()]
    
    if age_cert_filter == '+18':
        filtered_df = filtered_df[filtered_df['age_certification'] == '+18']
    if year_range:
        start_year= year_range
        filtered_df = filtered_df[(filtered_df['release_year'] >= start_year) ]
    num_filtered = filtered_df.shape[0]
    if num_filtered < 5:
        needed_rows = 5 - num_filtered
        additional_rows = df[~df.index.isin(filtered_df.index)]
        if not additional_rows.empty:
            additional_rows = additional_rows.sample(n=needed_rows, random_state=1)
            filtered_df = pd.concat([filtered_df, additional_rows], ignore_index=True)
    
    return filtered_df


def find_top_movies(df, input_list, top_n,number_of_users):
    weights = [0.7, 0.1, 0.2]
    input_embedding= torch.zeros_like(torch.tensor(df['combined_embedding'].iloc[0]))
    for input in input_list:
      embedding=torch.zeros_like(torch.tensor(df['combined_embedding'].iloc[0]))
      i=0
      for key,value in input.items():
        if key in ['genres','emotions','length']:
          embedding+=get_embedding(value)*weights[i]
        i+=1
      input_embedding+=embedding
    input_embedding/=number_of_users
    df['cosine_similarity'] = df['combined_embedding'].apply(lambda x: cosine_similarity(x.reshape(1, -1), input_embedding)[0][0])
    input=input_list[0]
    top_50_movies = df.nlargest(50, 'cosine_similarity')
    filtered_df= filter_df(top_50_movies,input['type'],input['age_certification'],input['release_year'])
    filtered_df=filtered_df.nlargest(min(10,len(filtered_df)), 'cosine_similarity')
    top_50_movies_sorted = filtered_df.sort_values(by='imdb_score', ascending=False)
    final_top_movies = top_50_movies_sorted.head(top_n)
    return final_top_movies['title'].tolist()
input = [{'genres':['action'],'emotions':['Sadness'],'length':'medium','type': 'movie','age_certification':'+18','release_year':1990},{'genres':['war','history'],'emotions':['Sadness'],'length':'medium','type': 'movie','age_certification':'+18','release_year':1990}]
top_movies = find_top_movies(df, input,3,2)
print(top_movies)

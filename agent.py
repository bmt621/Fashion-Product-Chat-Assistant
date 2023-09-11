import json
import cohere
import openai
import faiss
import numpy as np
import pandas as pd
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer
from keys import api_key,co_key



openai.api_key = api_key
co = cohere.Client(co_key)

class product_embedder:

  def __init__(self,catalog,embedding_path=None,is_sbert = False):

      self.catalog = catalog
      self.co = cohere.Client("emazTVHKTTrPot8n94AdSmGF70G3Q8cOazTsXSlO")
      self.threshold_score = 0.1
      self.is_bert = is_sbert

      if is_sbert:
        self.sbert_model = SentenceTransformer('all-mpnet-base-v2')

      if embedding_path is None:
        if is_sbert:
          self.embedding_data = self.sbert_embed_catalog()
        else:
          self.embedding_data = self.co_embed_catalog()
      else:
        self.embedding_data = pd.read_json(embedding_path,orient='records')

  def query(self,text):
    embeddings = np.array(self.embedding_data['embeddings'].to_list())

    
    q_embedding, d, I = self.faiss_search(embeddings,text)

    if len(q_embedding)==0:
       return None
    
    product_embeddings = np.array(self.embedding_data['embeddings'].to_list())
    cosine_sim = self.cosine_sim(I,product_embeddings,q_embedding)
    to_keep = [i for i in cosine_sim if i>self.threshold_score]

    n_keep = len(to_keep)
    if n_keep>0:
      new_idx = I[0][:n_keep]
      queries = self.bring_queries(new_idx)
      return queries.to_list()
    else:
      return False

  def bring_queries(self,idx):
    def extract_info_by_ids(row, target_ids):
      if row['id']-1 in target_ids:
          return row
      return None

    queries = self.catalog['products'].apply(lambda x: extract_info_by_ids(x,idx)).dropna()

    return queries


  def cosine_sim(self,I,all_embed,q_embed):

      if q_embed.shape[0] ==1:
        q_embed = q_embed.reshape(-1)

      embeddings = np.array([all_embed[i] for i in I[0][:len(all_embed)]])
      sim_score = np.dot(embeddings,q_embed)/(norm(embeddings,axis=1)*norm(q_embed))

      return sim_score

  def cohere_embed(self,all_texts):
        
        try:
           
            embeds = np.array(self.co.embed(texts=all_texts,model='large',truncate='LEFT').embeddings)
            return embeds
        
        except Exception as e:
           print("Exception {} occurs".format(str(e)))
           return np.array([])


  def sbert_embed(self,texts):
        embeddings = self.sbert_model.encode(texts)

        return embeddings


  def faiss_search(self,embeddings,text,top_k = 20):

        if self.is_bert:
          q_embed = self.sbert_embed(text)
        else:
          q_embed = self.cohere_embed([text])

        if len(q_embed)<=0:
           return [], [], []
           
        D=embeddings.shape[-1]
        index = faiss.IndexFlatIP(D)
        index.add(embeddings)

        d,i = index.search(q_embed.reshape(1,-1),top_k)

        return q_embed,d,i


  def sbert_embed_catalog(self):

    data = {"id":[],"embeddings":None}
    descriptions = []
    for i in range(len(self.catalog)):
      info = self.catalog['products'].iloc[i].copy()
      ID = info['id']
      data['id'].append(ID)
      del info['id']
      more_description = json.dumps(info)
      descriptions.append(more_description)

    desc_embeddings = self.sbert_embed(descriptions)
    data['embeddings'] = list(desc_embeddings)

    return pd.DataFrame(data)


  def gpt_response(self,inputs):

      instructions = """
      Instructions:
      based on this example generate your own response with the context of the example given below.
      1. User Query: "show me footwears for men."
        Engine Output:
        {
            'id': 18,
            'category': 'Footwear',
            'name': 'Sneakers',
            'brand': 'Reebok',
            'color': 'White',
            'size': '8',
            'price': 59.99,
            'description': 'Stylish white sneakers by Reebok, available in size 8.',
            'image_url': 'https://example.com/images/reebok_sneakers_white.jpg',
            'shop_url': 'https://example.com/shop/product/18'
        }
        assistant: "Here are some men's footwear options for you. Feel free to choose the one you like. If you still have trouble finding what you need, please search through the catalogs or leave us an email message."

      2. User Query: "tell me your name."
        Engine Output: "No products found."
        assistant: "I'm sorry, but I couldn't find any products matching your request. Could you please provide more information, or you can search through the catalogs or leave us a message."

      """


      re_try = True
      error = None
      output = None
      re_try_counter = 0

      while re_try:

        try:
            

            message = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[

                        {"role": "system", "content": "You are an intelligent search engine assistant for a product, tasked with providing helpful responses to user queries."},
                        {"role": "assistant", "content": instructions + "\n" + inputs},

                    ],
                temperature=0.7,
                max_tokens=1000,
                top_p=0.9,
                frequency_penalty=0.1,
                presence_penalty=1

                )
            outputs = message['choices'][0]['message']['content'].strip()

            return outputs
        
        except Exception as e:

            re_try_counter+=1
            if re_try_counter>3: # this will terminate loop and catch error message if we try for the third time and no luck.
               re_try = False
               return "exception {} occurs, please try calling this endpoint again.".format(str(e))
            print("Exception {}".format(str(e)))
            
        


  def co_embed_catalog(self):

    data = {"id":[],"embeddings":None}
    descriptions = []
    for i in range(len(self.catalog)):
      info = self.catalog['products'].iloc[i].copy()
      ID = info['id']
      data['id'].append(ID)
      del info['id']
      more_description = json.dumps(info)
      descriptions.append(more_description)

    desc_embeddings = self.cohere_embed(descriptions)
    data['embeddings'] = list(desc_embeddings)
    return pd.DataFrame(data)

  def chat_with_agent(self,query):

    outputs = self.query(query)

    if outputs:
      rand_len = len(outputs)
      randn = np.random.randint(0,rand_len)
      text = "User Query: {}, Engine Output: {}".format(query,outputs[randn])
      response = self.gpt_response(text)
    elif outputs==False:
      text = "User Query: {}, Engine Output: {}".format(query,"No Products Found")
      response = self.gpt_response(text)
    else:
       outputs = "Error when calling from database"
       response = "I cannot reached the database, please could you retry this url."

    

    return response,outputs
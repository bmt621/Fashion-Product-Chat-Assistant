from agent import product_embedder
import pandas as pd

catalog = pd.read_json('catalog.json')
agent = product_embedder(catalog)

print(agent.embedding_data)

agent.embedding_data.to_json('embeddings_with_ID.json',orient='records')

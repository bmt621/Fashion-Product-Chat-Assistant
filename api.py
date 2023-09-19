from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
from agent import product_embedder
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse


class Query(BaseModel):
    user_query:str
    

app = FastAPI()

catalog = pd.read_json('catalog.json')
agent = product_embedder(catalog,embedding_path='embeddings_with_ID.json')

@app.get('/')
def get_status():
    return {'status':"success"}

@app.post("/search_products")
def get_post_response(q:Query):
    query = q.user_query

    response, retrieved_data = agent.chat_with_agent(query)

    return {"response":response,'retrieved':retrieved_data}

  
@app.get('/{path:path}')
async def catch_all(path: str):
    return JSONResponse(content={"response": "GET Route not found", "retrieved": None}, status_code=404)

@app.post('/{path:path}')
async def catch_all_post(path: str):
    return JSONResponse(content={"response": "POST Route not found", "retrieved": None}, status_code=404)

if __name__ == "__main__":
    uvicorn.run(app,host='0.0.0.0',port=8000)
    

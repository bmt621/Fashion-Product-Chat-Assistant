#   Fashion-Product-Chatbot Application
# Public API Link [here](http://20.22.45.2/docs)

This API I created allows you to search for products in a catalog and received intelligent response and the retrieved items using natural language queries. It employs CoHERE, OpenAI's GPT-3, sentence-transformers and other libraries to provide intelligent responses to user queries. This Guide will show you step by step on how to run your own API on localhost, additionally, **I hosted a public API** [here](http://20.22.45.2/docs)

## Technologies and Frameworks
- python 3.10
- Docker
- kubernetes
- Pytorch
- FastAPI
- Azure Cloud
- Sentence-Transformers
- GPT-3
- Cohere Embedding Models
- Faiss (for searching through vector database)
- sentence BERT Model
- NVIDIA-CUDA

## Prerequisites

Before running the API, please make sure you have the following prerequisites installed:

- Python 3.10
- Docker (optional, for containerization)

## Installation Guide

1. Clone this repository to the local machine:

   ```bash
   git clone https://github.com/bmt621/Fashion-Product-Chat-Assistant.git
   ```
2. Navigate to the project directory:

   ```bash
   cd Fashion-Product-Chat-Assistant
   ```

3. Create a virtual environment (recommended):

   ```bash
   python -m venv venv
   ```

4. Activate the virtual environment:

   - On Windows:

     ```bash
     venv\Scripts\activate
     ```

   - On macOS and Linux:

     ```bash
     source venv/bin/activate
     ```

5. Install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

## Usage Instructions

1. Run the FastAPI application:

   ```bash
   python api.py
   ```

   - The API will be accessible at `http://localhost:8000` on your local machine.

2. To check the status of the API, open your web browser and navigate to:

   ```
   http://localhost:8000/
   ```

3. To search for products, you can use the `/search_products` endpoint. You can make a POST request with a JSON body containing your user query. For example:

   ```bash
   curl -X POST -H "Content-Type: application/json" -d '{"user_query": "show me shoes for women"}' http://localhost:8000/search_products
   ```

   Replace `"show me shoes for women"` with your desired user query.

4. The API will respond with an intelligent product recommendation based on your query and also a chatbot response showing you some product

## Docker (Optional)

If you prefer to run the API within a Docker container, the docker container will allows you to easily deploy on any cloud servers of your choice, in this implementation, I deployed the docker on azure cloud. Follow these additional steps to build and run the docker.

1. Build the Docker image from the project directory:

   ```bash
   docker build -t product-search-api .
   ```

2. But if you don't want to wait for the docker container to finish building, you can pull from here 👇
   ```bash
   docker pull orbitalsai/fashion-deployed
   ```

4. Run the Docker container:

   ```bash
   docker run -p 8000:8000 product-search-api
   ```

The API will be accessible at `http://localhost:8000` within the Docker container on your localhost machine.

## API Endpoint Documentation

This is the API Documentation.

### Base URL

The base URL for this API is `http://localhost:8000` when running locally or you can access the public one directly `http://20.22.45.2`, I also added kubernetes for efficient scaling up.

### Endpoints

#### 1. **Check API Status**

- **Endpoint**: `/`
- **HTTP Method**: GET
- **Description**: Check the status of the API to ensure it's running successfully.
- **Request Example**:

  ```bash
  GET http://localhost:8000/
  ```
  or

  ```bash
  GET http://20.22.45.2/docs
  ```

- **Response Example**:

  ```json
  {
      "status": "success"
  }
  ```

#### 2. **Search for Products**

- **Endpoint**: `/search_products`
- **HTTP Method**: POST
- **Description**: Search for products based on a user's natural language query. The API provides intelligent responses with product recommendations.
- **Request Example**:

  ```bash
  POST http://localhost:8000/search_products
  ```
  or
  ```bash
  POST http://20.22.45.2/search_products
  ```

  - **Request Body**:

    ```json
    {
        "user_query": "show me shoes for women"
    }
    ```

- **Response Example**:

  ```json
  {
      "response": "Here are some women's shoe options for you. Feel free to choose the one you like. If you still have trouble finding what you need, please search through the catalogs or leave us an email message.",
      "retrieved": [
          {
              "id": 18,
              "category": "Footwear",
              "name": "Stylish Women's Shoes",
              "brand": "FashionCo",
              "color": "Black",
              "size": "7",
              "price": 49.99,
              "description": "Fashionable black women's shoes by FashionCo, available in size 7.",
              "image_url": "https://example.com/images/fashionco_womens_shoes_black.jpg",
              "shop_url": "https://example.com/shop/product/18"
          },
          {
              "id": 22,
              "category": "Footwear",
              "name": "Casual Women's Sneakers",
              "brand": "SportyWear",
              "color": "White",
              "size": "6",
              "price": 39.99,
              "description": "Comfortable white women's sneakers by SportyWear, available in size 6.",
              "image_url": "https://example.com/images/sportywear_womens_sneakers_white.jpg",
              "shop_url": "https://example.com/shop/product/22"
          }
      ]
  }
  ```

### API Usage Instructions

1. Ensure that the API is running locally on machine or on public by checking its status at `http://localhost:8000/` or `http://20.22.45.2`

2. To search for products, make a POST request to `http://localhost:8000/search_products`. Provide a JSON request body with the user's query.

3. The API will respond with an intelligent product recommendation based on the user's query. The `response` field contains the API's reply, and the `retrieved` field contains a list of retrieved product recommendations.

4. You can modify the user query in the request body to search for different products.

### Swagger Documentation

For interactive testing, you can access Swagger UI by navigating to:

```
http://localhost:8000/docs
```
or access the public endpoint here 👇

```
http://20.22.45.2/docs
```

Swagger provides a user-friendly interface for exploring and interacting with the API endpoints.

---

The API documentation I created outlines the functionality and usage of the Product Search API, enabling users to search for products and receive intelligent responses based on their queries.


### NLU MODEL DOCUMENTATION
![sentence embedding](https://github.com/bmt621/Fashion-Product-Chat-Assistant/blob/main/img/embedding_example.jpg)
 
In the realm of intelligent search engines, a common methodology involves harnessing the power of transformer-based large language models. These models enable us to decipher the contextual nuances within sentences. The transformation of each sentence results in a unique embedding vector. In the diagram provided, these embeddings are represented as points, with each point corresponding to a sentence. The proximity of points in this vector space signifies shared context or meaning.

My approach takes this a step further by supplying the model with a comprehensive repository of product information. Each piece of product data is transformed into an embedding, which is then linked to a unique identification number (ID). You can explore a sample of this format in the JSON file located [here](https://github.com/bmt621/Fashion-Product-Chat-Assistant/blob/main/embeddings_with_ID.json). When a user submits a query, the model translates the query into a vector and evaluates its similarity to each product's embedding vector. Products whose embeddings closely align with the query's embedding receive higher similarity scores.

To refine our search results and eliminate noise, I employ the cosine similarity metric, augmented with a user-defined threshold. Products surpassing this threshold are deemed pertinent. Once we've identified these relevant documents, I employ GPT-3 to craft context-aware responses, enhancing the overall user experience. below is a simple picture depicting how vector similarity works. ![vector similarity](https://github.com/bmt621/Fashion-Product-Chat-Assistant/blob/main/img/vector-similarity-l.jpg)

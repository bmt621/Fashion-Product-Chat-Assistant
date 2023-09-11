import requests

url = 'http://0.0.0.0:8000/search_products'

# Define the data you want to send (if any) as a dictionary
data = {"user_query":"Show me a men's Polo T-shirt by Nike"}

response = requests.post(url, json=data)
response = response.json()

resp = response['response']
items = response['retrieved']

print("response: ",resp)
print("Items: ",items)
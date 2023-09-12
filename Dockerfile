FROM python:3.10.12

RUN apt-get update

RUN mkdir /Engine

WORKDIR /Engine


COPY requirements.txt requirements.txt

# Install packages - Use cache dependencies 
RUN pip install --no-cache-dir -r requirements.txt

# Copy our codes over to our working directory
COPY . .

EXPOSE 8000

# Run our project exposed on port 8080
CMD ["python3","api.py"]

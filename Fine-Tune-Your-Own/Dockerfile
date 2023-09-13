FROM nvcr.io/nvidia/pytorch:23.07-py3

RUN apt-get update

RUN mkdir /EngineTrainer

WORKDIR /EngineTrainer

COPY requirements.txt requirements.txt

# Install packages - Use cache dependencies 
RUN pip install --no-cache-dir -r requirements.txt

# Copy our codes over to our working directory
COPY . .

# Run our project exposed on port 8080
CMD ["python3","main.py"]
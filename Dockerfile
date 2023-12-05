FROM python:3.11.4

WORKDIR /Graduate_Thesis

COPY ./requirements.txt /Graduate_Thesis

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Install Java environment to run VnCoreNLP
RUN apt-get update && \
    apt-get install -y openjdk-11-jre-headless && \
    apt-get clean;

#COPY . /cta_matrix

# Run to download Automodel


CMD ["python", "main.py"]
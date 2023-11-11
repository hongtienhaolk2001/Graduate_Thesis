FROM python:3.11.4

WORKDIR /graduate_thesis

COPY ./requirements.txt /graduate_thesis

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Install Java environment to run VnCoreNLP
RUN apt-get update && \
    apt-get install -y openjdk-8-jre-headless && \
    apt-get clean;

COPY . /graduate_thesis

# Run to download Automodel
RUN python test.py

CMD ["python", "app.py"]
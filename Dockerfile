FROM python:3.11.4

WORKDIR /Graduate_Thesis

COPY ./requirements.txt /Graduate_Thesis

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Install Java environment to run VnCoreNLP
RUN apt-get update && \
    apt-get install -y default-jre && \
    apt-get clean;

COPY . /Graduate_Thesis

#EXPOSE 8080
#CMD ["gunicorn", "app:app", "-b", ":8080", "--timeout", "300"]
CMD ["python", "main.py"]
FROM python:3.11.4
# https://drive.google.com/file/d/13Qm55xy8sV6Tsu2WcKaGfNUSBeaeU8PO/view?usp=drive_link
COPY ./requirements.txt ./

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    apt-get update && \
    apt-get install -y openjdk-8-jre && \
    rm ./requirements.txt


# Copy local code to the container
COPY . /Graduate_Thesis


WORKDIR /Graduate_Thesis
EXPOSE 8080
CMD ["gunicorn", "main:app", "--timeout=0", "--preload", \
     "--workers=1", "--threads=4", "--bind=0.0.0.0:8080"]
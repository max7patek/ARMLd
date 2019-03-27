FROM ubuntu:latest
RUN apt-get update -y
RUN apt-get install -y python-pip python-dev build-essential
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
#CMD ["gunicorn", "-w 3", "server:app"]
ENTRYPOINT ["python"]
CMD ["server.py"]

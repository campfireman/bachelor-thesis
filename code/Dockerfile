FROM python:3.8.3-slim-buster

# install java for ABA-PRO
RUN sudo apt install default-jdk

COPY . /src

RUN pip install -r /src/requirements.txt
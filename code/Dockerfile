# FROM nvidia/cuda:11.4.2-runtime-ubuntu20.04

# # annoying ubuntu timezone stuff: https://dev.to/setevoy/docker-configure-tzdata-and-timezone-during-build-20bk
# ENV TZ=Europe/Berlin
# RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
FROM tensorflow/tensorflow:2.7.0-gpu
RUN mkdir /src

# install java for ABA-PRO
RUN apt-get update && \
    apt-get -y install python3.9 python3-pip git default-jre vim


COPY ./requirements.txt /src

RUN pip install -r /src/requirements.txt

COPY . /src

CMD ["python3", "/src/train.py"]
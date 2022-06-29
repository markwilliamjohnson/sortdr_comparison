FROM ubuntu:18.04
MAINTAINER Rafiq Saleh

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHON_VERSION=3.7
# install build utilities
RUN apt-get update && apt-get install -y libsm6 libxext6 libxrender-dev software-properties-common wget \
curl gnupg2 ca-certificates lsb-release

#installing python3.7
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update && apt-get install -y python3.7 python3-pip
RUN python3.7 -m pip install pip
RUN apt-get update && apt-get install -y python3-distutils python3-setuptools
RUN python3.7 -m pip install pip --upgrade pip

# Installing Ngnix latest stable version
RUN apt-get update
RUN echo "deb http://nginx.org/packages/ubuntu `lsb_release -cs` nginx" | tee /etc/apt/sources.list.d/nginx.list
RUN curl -fsSL https://nginx.org/keys/nginx_signing.key | apt-key add -
RUN apt-key fingerprint ABF5BD827BD9BF62
RUN apt-get update && apt-get install -y nginx
RUN apt-get clean

# Installing dependencies
RUN pip install --no-cache-dir requests==2.18.4
RUN pip install --no-cache-dir numpy==1.19.5
RUN pip install --no-cache-dir requests_oauthlib==1.3.0
RUN pip install --no-cache-dir oauthlib==3.1.0
RUN pip install --no-cache-dir pandas
RUN pip install --no-cache-dir openpyxl

# Installing extra python packages to run the inference code
RUN pip install flask gevent gunicorn && \
        rm -rf /root/.cache

ENV PYTHONUNBUFFERED=1

COPY src/ /opt/comparison/
WORKDIR /opt/comparison

EXPOSE 8080
# Running Python Application
CMD ["python3", "server.py"]
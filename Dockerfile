# Create docker image for the demo
#
# This docker image is based on existing notebook image
# It also includes the dependencies required for training and deploying
# this way we can use it as the base image
FROM gcr.io/kubeflow-images-public/tensorflow-1.12.0-notebook-cpu:v0.5.0

USER root

COPY requirements.txt .
RUN pip3 --no-cache-dir install -r requirements.txt

RUN apt-get update -y
RUN apt-get install -y emacs

USER jovyan
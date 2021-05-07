
# syntax=docker/dockerfile:1
FROM openvino/ubuntu18_dev:latest

USER root
RUN apt-get update && apt-get upgrade -y
RUN apt-get update && echo y | /opt/intel/openvino/install_dependencies/install_openvino_dependencies.sh
RUN apt-get update && apt-get install -y git

USER openvino
WORKDIR /home/openvino
RUN git clone https://github.com/dankirsdot/simple_tracker.git

# download model and convert it to IR
RUN python3 /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name yolo-v3-tiny-tf -o /home/openvino/ --cache /home/openvino/cache
RUN python3 /opt/intel/openvino/deployment_tools/tools/model_downloader/converter.py --name yolo-v3-tiny-tf

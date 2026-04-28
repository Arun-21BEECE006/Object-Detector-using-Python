#!/usr/bin/env bash

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Preparing YOLO files..."
mkdir -p yolo-coco

if [ ! -f yolo-coco/yolov3.weights ]; then
  echo "Downloading YOLO weights..."
  curl -L -o yolo-coco/yolov3.weights https://pjreddie.com/media/files/yolov3.weights
fi

echo "Build completed!"
# import numpy as np
# import argparse
# import cv2

# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True)
# ap.add_argument("-c", "--confidence", type=float, default=0.5)
# ap.add_argument("-t", "--threshold", type=float, default=0.3)
# args = vars(ap.parse_args())

# # load labels
# labelsPath = 'yolo-coco/coco.names'
# LABELS = open(labelsPath).read().strip().split("\n")

# COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

# weightsPath = 'yolo-coco/yolov3.weights'
# configPath = 'yolo-coco/yolov3.cfg'

# net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# # image = cv2.imread(args["image"])
# # (H, W) = image.shape[:2]/
# image = cv2.imread(args["image"])

# if image is None:
#     raise Exception(f"Error loading image at path: {args['image']}")

# (H, W) = image.shape[:2]

# ln = net.getUnconnectedOutLayersNames()

# blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
#                              swapRB=True, crop=False)
# net.setInput(blob)
# layerOutputs = net.forward(ln)

# boxes = []
# confidences = []
# classIDs = []

# for output in layerOutputs:
#     for detection in output:
#         scores = detection[5:]
#         classID = np.argmax(scores)
#         confidence = scores[classID]

#         if confidence > args["confidence"]:
#             box = detection[0:4] * np.array([W, H, W, H])
#             (centerX, centerY, width, height) = box.astype("int")

#             x = int(centerX - (width / 2))
#             y = int(centerY - (height / 2))

#             boxes.append([x, y, int(width), int(height)])
#             confidences.append(float(confidence))
#             classIDs.append(classID)

# idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"])

# if len(idxs) > 0:
#     for i in idxs.flatten():
#         (x, y) = (boxes[i][0], boxes[i][1])
#         (w, h) = (boxes[i][2], boxes[i][3])

#         color = [int(c) for c in COLORS[classIDs[i]]]
#         cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
#         text = "{}: {:.2f}".format(LABELS[classIDs[i]], confidences[i])
#         cv2.putText(image, text, (x, y - 5),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# # SAVE OUTPUT (IMPORTANT)
# output_path = "static/detected/output.jpg"
# cv2.imwrite(output_path, image)

# import numpy as np
# import cv2
# import os

# class YOLODetector:
#     def __init__(self):
#         labelsPath = "yolo-coco/coco.names"
#         weightsPath = "yolo-coco/yolov3.weights"
#         configPath = "yolo-coco/yolov3.cfg"

#         if not os.path.exists(weightsPath):
#             raise Exception("YOLO weights not found")

#         self.LABELS = open(labelsPath).read().strip().split("\n")
#         self.COLORS = np.random.randint(0, 255, size=(len(self.LABELS), 3), dtype="uint8")

#         print("Loading YOLO model...")
#         self.net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
#         self.ln = self.net.getUnconnectedOutLayersNames()

#     def detect(self, image_path, output_path, confidence=0.5, threshold=0.3):
#         image = cv2.imread(image_path)

#         if image is None:
#             raise Exception(f"Error loading image: {image_path}")

#         (H, W) = image.shape[:2]

#         blob = cv2.dnn.blobFromImage(
#             image, 1 / 255.0, (416, 416),
#             swapRB=True, crop=False
#         )

#         self.net.setInput(blob)
#         layerOutputs = self.net.forward(self.ln)

#         boxes = []
#         confidences = []
#         classIDs = []

#         for output in layerOutputs:
#             for detection in output:
#                 scores = detection[5:]
#                 classID = np.argmax(scores)
#                 conf = scores[classID]

#                 if conf > confidence:
#                     box = detection[0:4] * np.array([W, H, W, H])
#                     (centerX, centerY, width, height) = box.astype("int")

#                     x = int(centerX - (width / 2))
#                     y = int(centerY - (height / 2))

#                     boxes.append([x, y, int(width), int(height)])
#                     confidences.append(float(conf))
#                     classIDs.append(classID)

#         idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence, threshold)

#         if len(idxs) > 0:
#             for i in idxs.flatten():
#                 (x, y) = (boxes[i][0], boxes[i][1])
#                 (w, h) = (boxes[i][2], boxes[i][3])

#                 color = [int(c) for c in self.COLORS[classIDs[i]]]
#                 text = f"{self.LABELS[classIDs[i]]}: {confidences[i]:.2f}"

#                 cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
#                 cv2.putText(image, text, (x, y - 5),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

#         cv2.imwrite(output_path, image)

from ultralytics import YOLO
import cv2
import torch

class YOLODetector:
    def __init__(self):
        print("Loading YOLOv8 (optimized)...")

        # ⚠️ FORCE CPU MODE (VERY IMPORTANT for Render)
        torch.set_num_threads(1)

        self.model = YOLO("yolov8n.pt")

    def detect(self, image_path, output_path):

        # ⚠️ reduce memory usage
        results = self.model.predict(
            source=image_path,
            imgsz=320,   # LOWER resolution = LESS RAM
            conf=0.4,
            verbose=False
        )

        img = cv2.imread(image_path)

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])

                label = f"{self.model.names[cls]} {conf:.2f}"

                cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(img, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        cv2.imwrite(output_path, img)
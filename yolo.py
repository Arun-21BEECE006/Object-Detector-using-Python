import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True)
ap.add_argument("-c", "--confidence", type=float, default=0.5)
ap.add_argument("-t", "--threshold", type=float, default=0.3)
args = vars(ap.parse_args())

# load labels
labelsPath = 'yolo-coco/coco.names'
LABELS = open(labelsPath).read().strip().split("\n")

COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

weightsPath = 'yolo-coco/yolov3.weights'
configPath = 'yolo-coco/yolov3.cfg'

net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# image = cv2.imread(args["image"])
# (H, W) = image.shape[:2]/
image = cv2.imread(args["image"])

if image is None:
    raise Exception(f"Error loading image at path: {args['image']}")

(H, W) = image.shape[:2]

ln = net.getUnconnectedOutLayersNames()

blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                             swapRB=True, crop=False)
net.setInput(blob)
layerOutputs = net.forward(ln)

boxes = []
confidences = []
classIDs = []

for output in layerOutputs:
    for detection in output:
        scores = detection[5:]
        classID = np.argmax(scores)
        confidence = scores[classID]

        if confidence > args["confidence"]:
            box = detection[0:4] * np.array([W, H, W, H])
            (centerX, centerY, width, height) = box.astype("int")

            x = int(centerX - (width / 2))
            y = int(centerY - (height / 2))

            boxes.append([x, y, int(width), int(height)])
            confidences.append(float(confidence))
            classIDs.append(classID)

idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"])

if len(idxs) > 0:
    for i in idxs.flatten():
        (x, y) = (boxes[i][0], boxes[i][1])
        (w, h) = (boxes[i][2], boxes[i][3])

        color = [int(c) for c in COLORS[classIDs[i]]]
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        text = "{}: {:.2f}".format(LABELS[classIDs[i]], confidences[i])
        cv2.putText(image, text, (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# SAVE OUTPUT (IMPORTANT)
output_path = "static/detected/output.jpg"
cv2.imwrite(output_path, image)
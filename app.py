# from flask import Flask, render_template, request, jsonify
# import subprocess
# import os
# import time

# app = Flask(__name__)

# UPLOAD_FOLDER = 'static/images'
# OUTPUT_FOLDER = 'static/detected'

# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/about')
# def about():
#     return render_template('about.html')

# @app.route('/detect', methods=['POST'])
# def detect():
#     try:
#         image = request.files['image']

#         filename = f"input_{int(time.time())}.jpg"
#         # input_path = os.path.join(UPLOAD_FOLDER, filename)
#         input_path = os.path.abspath(os.path.join(UPLOAD_FOLDER, filename))
#         image.save(input_path)

#         # run YOLO
#         subprocess.run([
#             'python', 'yolo.py',
#             '--image', os.path.abspath(input_path).replace("\\", "/")
#         ], check=True)
#         # subprocess.run(['python', 'yolo.py', '--image', input_path], check=True)

#         output_path = 'static/detected/detected_image.jpg'
#         return jsonify({
#             'input_image': f'/{input_path}',
#             'output_image': f'/{output_path}?t={int(time.time())}'
#         })

#     except Exception as e:
#         return jsonify({'error': str(e)})

# if __name__ == '__main__':
#     app.run(debug=True)

# from flask import Flask, render_template, request, jsonify
# import os
# import time
# import subprocess

# app = Flask(__name__)

# UPLOAD_FOLDER = "static/uploads"
# OUTPUT_FOLDER = "static/detected"

# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# @app.route("/")
# def index():
#     return render_template("index.html")

# @app.route("/detect", methods=["POST"])
# def detect():
#     try:
#         file = request.files["image"]

#         filename = f"img_{int(time.time())}.jpg"
#         input_path = os.path.abspath(os.path.join(UPLOAD_FOLDER, filename))

#         file.save(input_path)

#         # Run YOLOv8
#         subprocess.run([
#             "python", "yolo.py",
#             "--image", input_path.replace("\\", "/")
#         ], check=True)
#         time.sleep(0.5)

#         return jsonify({
#             "input": f"/{UPLOAD_FOLDER}/{filename}",
#             "output": f"/static/detected/output.jpg?t={int(time.time())}"
#         })
#     except Exception as e:
#         return jsonify({"error": str(e)})

# if __name__ == "__main__":
#     app.run(debug=True)

# from flask import Flask, request, jsonify, render_template
# import os
# import time
# import requests
# import cloudinary
# import cloudinary.uploader
# from yolo import YOLODetector

# app = Flask(__name__)

# # Cloudinary config (SAFE way)
# cloudinary.config(
#     cloud_name=os.environ.get("CLOUDINARY_CLOUD_NAME"),
#     api_key=os.environ.get("CLOUDINARY_API_KEY"),
#     api_secret=os.environ.get("CLOUDINARY_API_SECRET"),
#     secure=True
# )

# # Temp folders (Render safe)
# UPLOAD_FOLDER = "temp"
# OUTPUT_FOLDER = "output"

# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# # Load YOLO once
# detector = YOLODetector()

# @app.route("/")
# def home():
#     return render_template("index.html")

# @app.route("/detect", methods=["POST"])
# def detect():
#     try:
#         if "image" not in request.files:
#             return jsonify({"error": "No image uploaded"}), 400

#         file = request.files["image"]

#         # Step 1: Upload original to Cloudinary
#         upload_result = cloudinary.uploader.upload(file)
#         input_url = upload_result["secure_url"]

#         # Step 2: Download for YOLO processing
#         img_data = requests.get(input_url).content
#         temp_input = os.path.join(UPLOAD_FOLDER, f"in_{int(time.time())}.jpg")

#         with open(temp_input, "wb") as f:
#             f.write(img_data)

#         # Step 3: Run YOLO
#         temp_output = os.path.join(OUTPUT_FOLDER, f"out_{int(time.time())}.jpg")
#         detector.detect(temp_input, temp_output)

#         # Step 4: Upload result image
#         output_upload = cloudinary.uploader.upload(temp_output)
#         output_url = output_upload["secure_url"]

#         return jsonify({
#             "input": input_url,
#             "output": output_url
#         })

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500


# if __name__ == "__main__":
#     port = int(os.environ.get("PORT", 8000))
#     app.run(host="0.0.0.0", port=port)

from flask import Flask, request, jsonify, render_template
import os
import time
import requests
import cloudinary
import cloudinary.uploader
from yolo import YOLODetector

app = Flask(__name__)

# Cloudinary config
cloudinary.config(
    cloud_name=os.environ.get("CLOUDINARY_CLOUD_NAME"),
    api_key=os.environ.get("CLOUDINARY_API_KEY"),
    api_secret=os.environ.get("CLOUDINARY_API_SECRET"),
    secure=True
)

UPLOAD_FOLDER = "temp"
OUTPUT_FOLDER = "output"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ⚠️ IMPORTANT: DO NOT load YOLO at startup (fixes Render crash)
detector = None

def get_detector():
    global detector
    if detector is None:
        print("Loading YOLO model (first request)...")
        detector = YOLODetector()
    return detector


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/detect", methods=["POST"])
def detect():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files["image"]

        # 1. Upload to Cloudinary
        upload_result = cloudinary.uploader.upload(file)
        input_url = upload_result["secure_url"]

        # 2. Download image locally
        img_data = requests.get(input_url).content

        input_path = os.path.join(UPLOAD_FOLDER, f"in_{int(time.time())}.jpg")
        with open(input_path, "wb") as f:
            f.write(img_data)

        # 3. Run YOLO (lazy-loaded)
        output_path = os.path.join(OUTPUT_FOLDER, f"out_{int(time.time())}.jpg")
        detector = get_detector()
        detector.detect(input_path, output_path)

        # 4. Upload result back to Cloudinary
        output_upload = cloudinary.uploader.upload(output_path)
        output_url = output_upload["secure_url"]

        return jsonify({
            "input": input_url,
            "output": output_url
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
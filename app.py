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

from flask import Flask, render_template, request, jsonify
import os
import time
import subprocess

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
OUTPUT_FOLDER = "static/detected"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/detect", methods=["POST"])
def detect():
    try:
        file = request.files["image"]

        filename = f"img_{int(time.time())}.jpg"
        input_path = os.path.abspath(os.path.join(UPLOAD_FOLDER, filename))

        file.save(input_path)

        # Run YOLOv8
        subprocess.run([
            "python", "yolo.py",
            "--image", input_path.replace("\\", "/")
        ], check=True)
        time.sleep(0.5)

        return jsonify({
            "input": f"/{UPLOAD_FOLDER}/{filename}",
            "output": f"/static/detected/output.jpg?t={int(time.time())}"
        })
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
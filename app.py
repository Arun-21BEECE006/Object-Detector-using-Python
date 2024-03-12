from flask import Flask, render_template, request, jsonify
import subprocess

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/detect', methods=['POST'])
def detect():
    # Save the uploaded image
    image = request.files['image']
    image_path = 'static/images/uploaclded_image.jpg'
    image.save(image_path)

    # Run the object detection script
    subprocess.run(['python', 'yolo.py', '--image', image_path])

    # Return the path to the output image
    output_image_path = 'static/images/detected_image.jpg'
    return jsonify({'output_image': output_image_path})

if __name__ == '__main__':
    app.run(debug=True)

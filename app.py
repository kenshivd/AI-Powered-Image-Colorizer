from flask import Flask, request, jsonify, send_file
import cv2
import numpy as np
import io
from PIL import Image

app = Flask(__name__)

# Load pre-trained Deep Learning colorization model
net = cv2.dnn.readNetFromCaffe("colorization_deploy_v2.prototxt", "colorization_release_v2.caffemodel")

@app.route("/colorize", methods=["POST"])
def colorize():
    file = request.files["file"]
    image = Image.open(io.BytesIO(file.read()))
    image = np.array(image)

    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l_channel = lab[:, :, 0]

    net.setInput(cv2.dnn.blobFromImage(l_channel))
    ab_channels = net.forward()
    ab_channels = ab_channels[0].transpose((1, 2, 0))

    lab[:, :, 1:] = ab_channels
    colorized_image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    _, buffer = cv2.imencode(".jpg", colorized_image)
    return send_file(io.BytesIO(buffer), mimetype="image/jpeg")

if __name__ == "__main__":
    app.run(debug=True)

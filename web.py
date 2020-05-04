
from flask import Flask, render_template, Response
from task import VideoCamera
import tensorflow as tf
import numpy as np

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Welcome to Hand Wash Monitoring System'

def gen():
    model = tf.keras.models.load_model("/home/raj/Hand-Hygiene-Monitoring-System-using-Gesture-Recognition/saved_model")
    predictor = VideoCamera(model=model)
    while True:
        frame = predictor.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
import tensorflow as tf
import cv2
import numpy as np 

class VideoCamera(object):
    def __init__(self, model):
        self.video = cv2.VideoCapture(0)
        self.model = model
        self.check = [0, 0, 0, 0, 0, 0]
        self.index = 0
        self.message = 'Start with step 1'
    
    def __del__(self):
        self.video.release()

    # returns camera frames along with bounding boxes and predictions
    def get_frame(self):
        _, image = self.video.read()
        image = cv2.resize(image,(256,256))
        image_norm = image/255.
        image_batch = np.expand_dims(image_norm, 0)
        
        pred = self.model.predict(image_batch)
        prediction = np.argmax(pred[0], axis=1)
        if prediction == self.index:
            if self.index != 6:
                self.check[self.index] += 1
                self.message = 'Doing Step' + str(self.index)
                if(self.check[self.index]>1):
                    self.index += 1
            else:
                self.message = 'All steps correctly followed'

        # return prediction
        cv2.putText(image, self.message + str(prediction), (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        _, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()

def main():
    model = tf.keras.models.load_model("/home/raj/Hand-Hygiene-Monitoring-System-using-Gesture-Recognition/saved_model")
    predictor = VideoCamera(model=model)
    while True:
        pred = predictor.get_frame()
        print(pred)

if __name__ == "__main__":
    main()




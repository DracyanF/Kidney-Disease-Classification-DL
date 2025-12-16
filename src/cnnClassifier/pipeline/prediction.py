import numpy as np
from tensorflow import keras
from keras.models import load_model
from keras.utils import load_img, img_to_array
import os



class PredictionPipeline:
    def __init__(self,filename):
        self.filename =filename


    
    def predict(self):
        # load model
        model = load_model(os.path.join("model", "model.h5"))

        imagename = self.filename
        test_image = load_img(imagename, target_size = (224,224))
        test_image = img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        prob = model.predict(test_image)[0][0]

        if prob >= 0.5:
            prediction = 'Normal'
            return [{ "image" : prediction}]
        else:
            prediction = 'Tumor'
            return [{ "image" : prediction}]
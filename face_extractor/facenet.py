from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from tensorflow import ConfigProto, Session
import numpy as np

class FacenetExtractor():
  def __init__(self, model_path="models/facenet_keras.h5"):
    self.model_path = model_path

  def init_model(self):
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    sess = Session(config=config)
    K.set_session(sess)
    self.model = load_model(self.model_path)

  def get_model(self):
    return self.model

  def get_embedding(self, face_image):
    face_image = face_image.astype("float32")
    mean, std = face_image.mean(), face_image.std()
    face_image = (face_image - mean) / std
    face_image = np.expand_dims(face_image, axis=0)
    y = self.model.predict(face_image)
    return y[0]
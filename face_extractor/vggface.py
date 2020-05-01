from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
import numpy as np

class VGGFaceExtractor():
  def __init__(self, model="senet50"):
    self.arch = model

  def init_model(self):
    self.model = VGGFace(model=self.arch, input_shape=(224, 224, 3), include_top=False, pooling="avg")

  def get_model(self):
    return self.model

  def get_embedding(self, face_image):
    face_image = face_image.astype("float32")
    face_image = np.expand_dims(face_image, axis=0)
    face_image = preprocess_input(face_image, version=1 if self.arch == "vgg16" else 2)
    y = self.model.predict(face_image)
    return y[0]
  
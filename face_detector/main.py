import cv2
from face_detector.model import TensoflowMobilNetSSDFaceDector

class Detector():
  def __init__(self, model_path='models/frozen_inference_graph_face.pb'):
    self.model = TensoflowMobilNetSSDFaceDector(model_path)

  def detect(self, image):
    boxes = self.model.detect_face(image)
    cropped_images = []
    for box in boxes:
      cropped_image = image[box[1]:box[3], box[0]:box[2]]
      cropped_images.append(cropped_image)
    return cropped_images, boxes
    
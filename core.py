from dotenv import load_dotenv
load_dotenv()

import cv2
import numpy as np
import os
import time

from db.connection import DbConnection
from db.identity import Identity
from db.logtime import Logtime
from face_extractor.facenet import FacenetExtractor
from super_resolution.main import SuperResolution
from face_detector.main import Detector
from utils import Utils
from siamese import Siamese

class Core():
  def __init__(self):
    self.super_resolution = SuperResolution(model_path=os.getenv("SUPER_RESOLUTION_DIR"))
    self.detector = Detector(os.getenv("DETECTOR_DIR"))
    self.extractor = FacenetExtractor(model_path=os.getenv("EXTRACTOR_DIR"))
    self.extractor.init_model()
    self.utils = Utils(
      super_resolution=self.super_resolution,
      extractor=self.extractor
    )
    self.classifier = Siamese()
    self.classifier.load_model(os.getenv("CLASSIFIER_DIR"))
    self.db = DbConnection()

  def save_embedding(self, image, name):
    embedding = self.utils.extract_embedding(image)
    identity = Identity(name, embedding.tolist())
    self.db.insert_identity(vars(identity))
    print("Successfully save face embedding to database")

  def find_identity(self, image):
    embedding = self.utils.extract_embedding(image)
    identities = []
    for identity in self.db.get_identities():
      i_embedding = np.asarray(identity["face_embedding"])
      prediction = self.classifier.predict([embedding.reshape(1, 128), i_embedding.reshape(1, 128)])
      identities.append({
        "_id": identity["_id"],
        "name": identity["name"],
        "prediction": prediction
      })
    identities = list(filter(lambda el: el["prediction"] > 0.5, identities))
    if len(identities) == 0:
      return None
    recognized_identity = max(identities, key=lambda i: i["prediction"])
    return recognized_identity
  
  def one_shot_learning(self, image_path=None, identity_name=""):
    if (image_path):
      image = cv2.imread(image_path)
      cropped_images, _ = self.detector.detect(image)
      if (len(cropped_images)):
        cropped_image = cropped_images[0]
        self.save_embedding(cropped_image, identity_name)
      else:
        print("No face is found")
        return
    else:
      cap = cv2.VideoCapture(0)
      while(True):
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(1)
        detected_faces, _ = self.detector.detect(frame)
        if (len(detected_faces)):
          detected_face = detected_faces[0]
          self.save_embedding(detected_face, identity_name)
          break
      
      cap.release()
      cv2.destroyAllWindows()

  def save_timestamp(self, identities, threshold=None):
    current_t = time.time()
    identities_id = [identity["_id"] for identity in identities if identity is not None]
    identities_id = list(set(identities_id))
    for logtime in self.db.get_logtimes({"left_time": None}):
      if logtime["identity_id"] not in identities_id:
        self.db.set_logtime({"_id": logtime["_id"]}, {"left_time": current_t})
      else:
        identities_id.remove(logtime["identity_id"])

    for logtime in self.db.get_logtimes({"left_time": {"$gte": current_t - threshold}}):
      if logtime["identity_id"] in identities_id:
        self.db.set_logtime({"_id": logtime["_id"]}, {"left_time": None})
        identities_id.remove(logtime["identity_id"])

    for identity_id in identities_id:
      log_time = Logtime(identity_id, current_t, None)
      self.db.insert_logtime(vars(log_time))

  def recognition_summary(self, start_time, end_time):
    self.db.set_logtime({"left_time": None}, {"left_time": end_time})
    print("=== Recognition Summary ===")
    for logtime in self.db.get_logtimes({"enter_time": {"$gte": start_time}, "left_time": {"$lte": end_time}}):
      identity = self.db.get_identities(opt={"_id": logtime["identity_id"]}).next()
      print("{} ({}). start:{}. end:{}.".format(
        identity["name"], 
        identity["_id"], 
        time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(logtime["enter_time"])), 
        time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(logtime["left_time"]))
      ))

  def recognition_log(self, threshold, video_path=None):
    cap = cv2.VideoCapture(video_path or 0)
    start_time = time.time()
    while(True if not video_path else cap.isOpened()):
      _, frame = cap.read()
      faces, boxes = self.detector.detect(frame)
      identities = []
      for face in faces:
        identity = self.find_identity(face)
        identities.append(identity)

      self.save_timestamp(identities, threshold)

      for idx, box in enumerate(boxes):
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255,0,0), 2)
        if (not identities[idx]):
          name = "Unidentified"
        else:
          name = identities[idx]["name"]
        cv2.putText(frame, name, (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
      cv2.imshow('frame', frame)
      key = cv2.waitKey(30)
      if key == 27:
        break
    end_time = time.time()
    self.recognition_summary(start_time, end_time)

if __name__ == "__main__":
  core = Core()
  # image = cv2.imread("allison_janney2.jpg")
  # faces, _ = core.detector.detect(image)
  # image = faces[0]
  # print(core.find_identity(image))
  # core.one_shot_learning(identity_name="Allison Janney", image_path="allison_janney2.jpg")
  core.recognition_log()
    

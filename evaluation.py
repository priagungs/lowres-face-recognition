from dotenv import load_dotenv
load_dotenv()

from face_detector.main import Detector
from super_resolution.main import SuperResolution
from utils import Utils
from face_extractor.facenet import FacenetExtractor
from siamese import Siamese

import cv2
import numpy as np
import os
from tqdm import tqdm
import time

class Evaluation():
  def __init__(self, dataset_path="datasets/lfw-dataset", img_size=(36, 36)):
    self.super_resolution = SuperResolution(model_path=os.getenv("SUPER_RESOLUTION_DIR"))
    self.detector = Detector(os.getenv("DETECTOR_DIR"))
    self.face_extractor = FacenetExtractor(model_path=os.getenv("EXTRACTOR_DIR"))
    self.utils = Utils(super_resolution=self.super_resolution)
    self.dataset_path = dataset_path
    self.input_size = (160, 160)
    self.img_size = img_size
    self.face_extractor.init_model()
    self.classifier = Siamese()
    self.classifier.load_model(os.getenv("CLASSIFIER_DIR"))

  def evaluate(
    self,
    match_test="matchpairsDevTest.csv",
    mismatch_test="mismatchpairsDevTest.csv",
  ):
    match_test = np.genfromtxt(os.path.join(self.dataset_path, match_test), delimiter=",", dtype=None, skip_header=1)
    mismatch_test = np.genfromtxt(os.path.join(self.dataset_path, mismatch_test), delimiter=",", dtype=None, skip_header=1)

    times1, accuracies1 = self.evaluate_data(match_test, True)
    times2, accuracies2 = self.evaluate_data(mismatch_test, False)

    print("Average time {}".format(np.mean(times1 + times2)))
    print("Average accuracies {}".format(np.mean(accuracies1 + accuracies2)))

  def write_undetected(self, image, imagename, method="mtcnn"):
    path = os.path.join(os.getcwd(), "results/undetected/{}".format(method), imagename)
    cv2.imwrite(path, image)

  def write_result(self, image1, image2, imagename, is_match, size="12x12", result="unrecognized"):
    if (is_match):
      path = os.path.join(os.getcwd(), "results/{}/match".format(result), size, imagename)
    else:
      path = os.path.join(os.getcwd(), "results/{}/mismatch".format(result), size, imagename)
    vis = np.concatenate((image1, image2), axis=0)
    cv2.imwrite(path, vis)

  def evaluate_data(self, dataset, is_match):
    times = []
    accuracies = []

    for idx, data in tqdm(enumerate(dataset), total=len(dataset)):
      path1, path2 = self.get_images_path(data, is_match)

      uncropped_image1 = cv2.imread(path1)
      uncropped_image2 = cv2.imread(path2)
      image1, image2 = uncropped_image1, uncropped_image2

      execution_time1 = 0
      execution_time2 = 0
      start_time = time.time()
      try:
        image1 = self.detector.detect(uncropped_image1)[0][0]
        image2 = self.detector.detect(uncropped_image2)[0][0]
      except IndexError:
        if (image1.shape == uncropped_image1.shape):
          self.write_undetected(uncropped_image1, os.path.basename(path1), method="mtcnn")
        if (image2.shape == uncropped_image2.shape):
          self.write_undetected(uncropped_image2, os.path.basename(path2), method="mtcnn")
        accuracies.append(0)
        continue
      
      execution_time1 += (time.time() - start_time)
      execution_time2 += (time.time() - start_time)

      sr_1 = cv2.resize(image1, self.img_size, interpolation=cv2.INTER_CUBIC)
      
      start_time = time.time()
      sr_1 = self.utils.resize_image(sr_1, size=self.input_size)
      execution_time1 += (time.time() - start_time)

      sr_2 = cv2.resize(image2, self.img_size, interpolation=cv2.INTER_CUBIC)

      sample_image2 = cv2.resize(sr_2, self.input_size, interpolation=cv2.INTER_CUBIC)

      start_time = time.time()
      sr_2 = self.utils.resize_image(sr_2, size=self.input_size)
      execution_time2 += (time.time() - start_time)

      start_time = time.time()
      sr_1 = self.face_extractor.get_embedding(sr_1)
      execution_time1 += (time.time() - start_time)

      start_time = time.time()
      sr_2 = self.face_extractor.get_embedding(sr_2)
      execution_time2 += (time.time() - start_time)

      image1 = self.utils.resize_image(image1, size=self.input_size)
      image2 = self.utils.resize_image(image2, size=self.input_size)

      sample_image1 = np.copy(image1)

      image1 = self.face_extractor.get_embedding(image1)
      image2 = self.face_extractor.get_embedding(image2)

      classifier = self.classifier.get_model()
      
      start_time = time.time()
      result1 = classifier.evaluate([sr_1.reshape(1, 128), image2.reshape(1, 128)], [1 if is_match else 0])
      execution_time1 += (time.time() - start_time)

      start_time = time.time()
      result2 = classifier.evaluate([sr_2.reshape(1, 128), image1.reshape(1, 128)], [1 if is_match else 0])
      execution_time2 += (time.time() - start_time)

      if (result2[1] < 0.5):
        self.write_result(sample_image1, sample_image2, "{}.jpg".format(idx+1), is_match, size="36x36", result="unrecognized")
      else:
        self.write_result(sample_image1, sample_image2, "{}.jpg".format(idx+1), is_match, size="36x36", result="recognized")

      accuracies.append(result1[1])
      accuracies.append(result2[1])
      
      times.append(execution_time1)
      times.append(execution_time2)

    return times, accuracies

  def get_images_path(self, pair, is_match):
    if (is_match):
      image_dir = os.path.join(self.dataset_path, "lfw-deepfunneled", pair[0].astype("str"))
      image_list = os.listdir(image_dir)
      return os.path.join(image_dir, image_list[pair[1] - 1]), os.path.join(image_dir, image_list[pair[2] - 1])
    else:
      image_dir1 = os.path.join(self.dataset_path, "lfw-deepfunneled", pair[0].astype("str"))
      image_list1 = os.listdir(image_dir1)

      image_dir2 = os.path.join(self.dataset_path, "lfw-deepfunneled", pair[2].astype("str"))
      image_list2 = os.listdir(image_dir2)

      return os.path.join(image_dir1, image_list1[pair[1] - 1]), os.path.join(image_dir2, image_list2[pair[3] - 1])

if __name__ == "__main__":
  evaluation = Evaluation()
  evaluation.evaluate()
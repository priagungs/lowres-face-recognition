from dotenv import load_dotenv
load_dotenv()

from face_detector.main import Detector
from super_resolution.main import SuperResolution
from utils import Utils
from face_extractor.facenet import FacenetExtractor
from face_extractor.vggface import VGGFaceExtractor
import cv2
import numpy as np
import os
from tqdm import tqdm

class DatasetGenerator():
  def __init__(self, arch="facenet", dataset_path="datasets/lfw-dataset"):
    self.super_resolution = SuperResolution(model_path=os.getenv("SUPER_RESOLUTION_DIR"))
    self.detector = Detector(os.getenv("DETECTOR_DIR"))
    self.arch = arch
    self.face_extractor = FacenetExtractor(model_path=os.getenv("EXTRACTOR_DIR")) if arch == "facenet" else VGGFaceExtractor(model=arch)
    self.utils = Utils(super_resolution=self.super_resolution)
    self.dataset_path = dataset_path
    self.input_size = (160, 160) if arch == "facenet" else (224, 224)
    self.face_extractor.init_model()
  

  def generate(
    self,
    match_test="matchpairsDevTest.csv",
    match_train="matchpairsDevTrain.csv",
    mismatch_test="mismatchpairsDevTest.csv",
    mismatch_train="mismatchpairsDevTrain.csv"
  ):
    match_test = np.genfromtxt(os.path.join(self.dataset_path, match_test), delimiter=",", dtype=None, skip_header=1)
    match_train = np.genfromtxt(os.path.join(self.dataset_path, match_train), delimiter=",", dtype=None, skip_header=1)
    mismatch_test = np.genfromtxt(os.path.join(self.dataset_path, mismatch_test), delimiter=",", dtype=None, skip_header=1)
    mismatch_train = np.genfromtxt(os.path.join(self.dataset_path, mismatch_train), delimiter=",", dtype=None, skip_header=1)
    
    match_test = self.extract_data(match_test, True)
    print(match_test.shape)

    match_train = self.extract_data(match_train, True)
    print(match_train.shape)

    mismatch_test = self.extract_data(mismatch_test, False)
    print(mismatch_test.shape)

    mismatch_train = self.extract_data(mismatch_train, False)
    print(mismatch_train.shape)

    x_train = np.concatenate((match_train, mismatch_train))
    x_test = np.concatenate((match_test, mismatch_test))

    print("X_TRAIN SHAPE")
    print(x_train.shape)
    print("X_TEST SHAPE")
    print(x_test.shape)

    y_test = np.append(np.full(match_test.shape[0], 1), np.full(mismatch_test.shape[0], 0))
    print("Y_TEST")
    print(y_test)

    y_train = np.append(np.full(match_train.shape[0], 1), np.full(mismatch_train.shape[0], 0))
    print("Y_TRAIN")
    print(y_train)

    np.savez_compressed("lfw-pair-test-senet50.npz", x_train, y_train, x_test, y_test)


  def extract_data(self, datalist, is_match):
    result = []
    for data in tqdm(datalist, total=len(datalist)):
      image1, image2 = self.get_images_path(data, is_match)

      image1 = cv2.imread(image1)
      image2 = cv2.imread(image2)

      try:
        image1 = self.detector.detect(image1)[0][0]
        image2 = self.detector.detect(image2)[0][0]
      except IndexError:
        continue
      
      sr_1 = cv2.resize(image1, (36, 36), interpolation=cv2.INTER_CUBIC)
      sr_1 = self.utils.resize_image(sr_1, size=self.input_size)

      sr_2 = cv2.resize(image2, (36, 36), interpolation=cv2.INTER_CUBIC)
      sr_2 = self.utils.resize_image(sr_2, size=self.input_size)

      image1 = self.utils.resize_image(image1, size=self.input_size)
      image2 = self.utils.resize_image(image2, size=self.input_size)
      
      result.append(np.stack((self.face_extractor.get_embedding(image1), self.face_extractor.get_embedding(sr_2))))
      result.append(np.stack((self.face_extractor.get_embedding(sr_1), self.face_extractor.get_embedding(image2))))
    return np.array(result)


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


  def start(self):
    self.generate()

if __name__ == "__main__":
  DatasetGenerator(arch="senet50").start()


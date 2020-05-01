import cv2
import math

class Utils():
  def __init__(self, super_resolution=None, extractor=None):
    self.super_resolution = super_resolution
    self.extractor = extractor

  def resize_image(self, image, size=(160, 160)):
    h, w, _ = image.shape
    if ((h, w) == size):
      return image
    elif (h > size[0] and w > size[1]):
      return cv2.resize(image, size, interpolation = cv2.INTER_AREA)
    elif (h < size[0] and h < w):
      image = cv2.resize(image, (h, math.floor(h * (size[1] / size[0]))), interpolation = cv2.INTER_AREA)
    elif (w < size[1] and w < h):
      image = cv2.resize(image, (w, math.floor(w * (size[1] / size[0]))), interpolation = cv2.INTER_AREA)
    
    return self.upscale_image(image, size)
  
  def upscale_image(self, image, size):
    h, _, _ = image.shape
    scale = (size[0] // h)
    scale = scale + 1 if (size[0] - (h * scale)) > (h * (scale + 1) - size[0]) else scale

    if scale <= 4 and scale > 1:
        sr_image = self.super_resolution.execute(image, scale)
    else:
      remain_scale = scale
      sr_image = image
      while (remain_scale // 4 > 0):
        sr_image = self.super_resolution.execute(sr_image, 4)
        remain_scale //= 4
      while (remain_scale // 3 > 0):
        sr_image = self.super_resolution.execute(sr_image, 3)
        remain_scale //= 3
      while (remain_scale // 2 > 0):
        sr_image = self.super_resolution.execute(sr_image, 2)
        remain_scale //= 2

    if (sr_image.shape == size):
      return sr_image
    else:
      return cv2.resize(sr_image, size, interpolation = cv2.INTER_AREA)

  def extract_embedding(self, image, input_size=(160, 160)):
    face_image = self.resize_image(image, size=input_size)
    return self.extractor.get_embedding(face_image)

import torch
import torchvision.transforms as transforms
import cv2

from collections import OrderedDict
from super_resolution import model

class SuperResolution():
  def __init__(self, model_path="models/carn.pth"):
    self.net = model.Net(multi_scale=True)

    state_dict = torch.load(model_path)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
      name = k
      new_state_dict[name] = v
    
    self.net.load_state_dict(new_state_dict)
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.net = self.net.to(self.device)
    self.transform = transforms.Compose([
      transforms.ToTensor()
    ])

  def execute(self, lr_image, scale):
    with torch.no_grad():
      lr_image = self.transform(lr_image)
      lr_image = lr_image.unsqueeze(0).to(self.device)
      sr_image = self.net(lr_image, scale).detach().squeeze(0)
      lr_image = lr_image.squeeze(0)

    sr_image = sr_image.cpu()
    sr_image = sr_image.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()

    return sr_image
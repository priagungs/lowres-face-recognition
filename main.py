import argparse
from core import Core

parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=int, help="app mode (1: one shot learning, 2: recognition)",
  choices=[1, 2])
parser.add_argument("--source", help="source (camera, video)",
  choices=["camera", "video"])
parser.add_argument("--source_dir", help="video/image dir")
parser.add_argument("--identity_name", help="identity name")
parser.add_argument("--threshold", type=float, help="Threshold time in second to count between two recognition as one logtime")
args = parser.parse_args()

core = Core()

if (args.mode == 1):
  if (not args.identity_name):
    print("--identity_name must be filled")
  else:
    core.one_shot_learning(image_path=args.source_dir, identity_name=args.identity_name)
else:
  if (args.threshold):
    if (args.source == "video" and args.source_dir):
      core.recognition_log(args.threshold, video_path=args.source_dir)
    else: 
      core.recognition_log(args.threshold)
  else:
    print("--threshold must be filled")

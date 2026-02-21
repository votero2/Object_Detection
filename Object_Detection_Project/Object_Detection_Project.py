from cProfile import label
import cv2
from src import split_dataset
from src.camera import Camera
from src.capture import run_capture
import sys


if len(sys.argv) < 2:
    print("Usage: python Object_Detection_Project.py <label>")
    print("Example: python Object_Detection_Project.py cup")
    raise SystemExit(1)

label = sys.argv[1]
run_capture(label)


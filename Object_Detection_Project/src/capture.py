from pathlib import Path
import time
import cv2
from src.camera import Camera


def run_capture(label: str):
  PROJECT_ROOT = Path(__file__).resolve().parents[1]
  SAVE_DIR = PROJECT_ROOT / "data" / "images" /label
  SAVE_DIR.mkdir(parents = True, exist_ok = True)

  print("Saving images to: ", SAVE_DIR)
  cam = Camera() # force iriun index

  count = 0
  print("CLICK the video window first, then press:")
  print("  S = save image")
  print("  Q = quit")

  while True:
      frame = cam.read()
      if frame is None:
         print("No frame received.")
         break

      cv2.imshow("Capture (click here, then press S to save)", frame)

      key = cv2.waitKey(1) & 0xFF

      if key in (ord('s'), ord('S')): # spacebar
         filename = SAVE_DIR / f"img_{int(time.time())}_{count}.jpg"
         ok = cv2.imwrite(str(filename),frame)
         print("Saved" if ok else "Failed to save", filename)
         count += 1

      if key in (ord('q'), ord('Q')):
         break


  cam.release()
  cv2.destroyAllWindows()






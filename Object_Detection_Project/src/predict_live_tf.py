import json
from tkinter import Label
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
from camera import Camera
from helper.draw_auto_box import draw_auto_box




HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

MODEL_PATH = HERE / "tf_model.keras"
LABEL_PATH = HERE / "labels.json"

if not LABEL_PATH.exists():
    raise FileNotFoundError(f"labels.json not found at: {LABEL_PATH}")



print("MODEL_PATH =", MODEL_PATH)
print("LABEL_PATH =", LABEL_PATH)


#load labels
with open(LABEL_PATH,"r",encoding= "utf-8") as f:
    class_names = json.load(f)

if isinstance(class_names, dict):
    class_names = [class_names[str(i)] for i in range(len(class_names))]

#load models
model = tf.keras.models.load_model(MODEL_PATH)

#camera used(iPhone)
cam = Camera(index=1)

# settings
IMG_SIZE = (224, 224)
CONF_THRESHOLD = 0.25
MIN_CONF = 0.60

prev_box = None
while True:
    frame = cam.read()
    if frame is None:
       break
    
    #create img from frame
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)

    #Prepare tensor
    x = np.expand_dims(img, axis= 0).astype(np.float32)

    #match training preproc
    # x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    #Must match training
    #x = (x/127.5) -1.0

    #Predict
    preds = model.predict(x, verbose= 0)[0]
    class_id = int(np.argmax(preds))
    confidence = float(preds[class_id])
    label = class_names[class_id]

    text = ""
    if confidence < MIN_CONF:
        label_to_show = "Unknown"
    else:
        label_to_show = label

    if confidence >= CONF_THRESHOLD:
        text = f"{label} {confidence:.2f}"
    else:
        text = f"Unknown {confidence:.2f}"

    cv2.putText(frame,text, (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2)

   
    frame, box = draw_auto_box(frame)

     #smooth rectangle
    if box is not None:
        if prev_box is None:
            prev_box = box
        else:
            alpha = 0.85 #smothing factor
            prev_box = tuple(
                int(alpha * p + (1 - alpha) * b)
                for p,b in zip(prev_box,box)
                )
        x,y,w,h = prev_box
        cv2.rectangle(frame, (x,y) , (x+w, y+h), (0,255,0),2)
   
   
    cv2.imshow("Predict Live", frame)
    
    key = cv2.waitKey(1) & 0xFF

    if key in (ord('q'), ord('Q')):
         break

cam.release()
cv2.destroyAllWindows()



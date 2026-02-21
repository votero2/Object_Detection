from mailbox import NoSuchMailboxError
from tabnanny import verbose
import tkinter as tk
from tkinter import ttk
from token import COMMA
import cv2
from cv2.gapi.wip import draw
import numpy as np
from PIL import Image , ImageTk
import tensorflow as tf
import json
from pathlib import Path
from camera import Camera

#import auto_box
try:
    from helper.draw_auto_box import draw_auto_box
except Exception:
    draw_auto_box = None

# Config
HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "tf_model.keras"
LABEL_PATH = HERE / "labels.json"
IMG_SIZE = (224,224)

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Live Object Classifier (TF)")
        self.geometry("980x640")

        #load labels
        with open(LABEL_PATH, "r", encoding="utf-8") as f:
            labels_obj = json.load(f)

        #labels.json may be list or dict
        if isinstance(labels_obj,dict):
            self.class_names = [labels_obj[str(i)] for i in range(len(labels_obj))]
        else:
            self.class_names = labels_obj

        #load models
        self.model = tf.keras.models.load_model(MODEL_PATH)

        #state
        self.cap = None
        self.running = False
        self.prev_box = None

        #UI vars
        self.camera_index = tk.IntVar(value = 1)
        self.conf_threshold =tk.DoubleVar(value=0.60)
        self.use_box = tk.BooleanVar(value=True)

        self._build_ui()


    def _build_ui(self):
        #Left: video
        self.video_label = ttk.Label(self)
        self.video_label.pack(side=tk.LEFT, padx=10, pady=10)

        #right: controls
        right = ttk.Frame(self)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True,padx=10,pady= 10)

        ttk.Label(right, text="Controls", font=("Segoe UI", 14, "bold")).pack(anchor="w",pady=(0,10))

        #Camera index
        cam_row = ttk.Frame(right)
        cam_row.pack(fill=tk.X,pady=5)
        ttk.Label(cam_row, text="Camera Index:").pack(side=tk.LEFT)
        ttk.Spinbox(cam_row, from_= 0, to=10, textvariable=self.camera_index, width=5).pack(side=tk.LEFT,padx=8)

        #Threshold slider
        ttk.Label(right, text="Confidence Threshold:").pack(anchor="w",pady=(12,0))
        ttk.Scale(right, from_=0.0, to=1.0, variable=self.conf_threshold).pack(fill=tk.X, pady=5)
        self.thresh_readout = ttk.Label(right, text= "0.60")
        self.thresh_readout.pack(anchor="w")

        #Box toggle
        ttk.Checkbutton(
            right,
            text = "Auto-box (contour heuristic)",
            variable=self.use_box
            ).pack(anchor="w", pady=(12,0))

        #Start/Stop buttons
        btn_row = ttk.Frame(right)
        btn_row.pack(fill=tk.X, pady=15)
        ttk.Button(btn_row, text="Start", command=self.start).pack(side=tk.LEFT,padx= 5)
        ttk.Button(btn_row, text="Stop", command=self.stop).pack(side=tk.LEFT, padx=5)

        #Prediction readout
        ttk.Separator(right).pack(fill=tk.X, pady=10)
        ttk.Label(right, text="Prediction:", font=("Segoe UI",12,"bold")).pack(anchor="w")
        self.pred_label = ttk.Label(right, text="-",font=("Segoe UI", 16))
        self.pred_label.pack(anchor="w",pady=(6,0))

        self.conf_label = ttk.Label(right, text="", font=("Segoe UI",11))
        self.conf_label.pack(anchor="w",pady=(2,0))

        ttk.Label(
            right,
            text="Tip: press Stop before closing the window.",
            foreground="gray"
            ).pack(anchor="w",pady=(20,0))

        # Update threshold display regularly
        self.after(100, self._update_threshold_text)

        #Close handler
        self.protocol("WM_DELETE_WINDOW", self.on_close)


    def _update_threshold_text(self):
        self.thresh_readout.config(text=f"{self.conf_threshold.get():.2f}")
        self.after(100,self._update_threshold_text)


    def start(self):
        if self.running:
            return

        idx = int(self.camera_index.get())
        self.cap = Camera(index= 1)

        self.running = True
        self.prev_box = None
        self._loop()

    def stop(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.prev_box = None

    def _predict(self, frame_bgr):
        #Resize for model
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img_rgb = cv2.resize(img_rgb, IMG_SIZE)

        x = np.expand_dims(img_rgb, axis=0).astype(np.float32)
        
        preds = self.model.predict(x, verbose=0)[0]
        class_id = int(np.argmax(preds))
        confidence = float(preds[class_id])
        label = self.class_names[class_id]

        return label, confidence, preds

    def _smooth_box(self, box):
        if box is None:
            self.prev_box = None
            return None

        if self.prev_box is None:
            self.prev_box = box
            return box

        alpha = 0.85
        self.prev_box = tuple(
            int(alpha * p + (1 - alpha) * b)
            for p, b in zip(self.prev_box, box)
            )
        return self.prev_box


    def _loop(self):
        if not self.running or self.cap is None:
            return

        frame =self.cap.read()
        
        #predict
        label, confidence, _ = self._predict(frame)
        thr = float(self.conf_threshold.get())

        if confidence >= thr:
            shown = label
        else:
            shown = "Unknown"

        self.pred_label.config(text=shown)
        self.conf_label.config(text=f"Confidence: {confidence:.2f} (threshold {thr: .2f})")

        #Draw label on frame
        cv2.putText(frame, f"{shown} {confidence:.2f}",
                    (20,40), cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

        if self.use_box.get() and draw_auto_box:
            frame2, box = draw_auto_box(frame.copy())
            smoothed = self._smooth_box(box)
            frame = frame2

            if smoothed:
                x, y, w, h = smoothed
                cv2.rectangle(frame, (x,y),(x+ w, y + h),(0,255,0),2)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb).resize((640, 480))
        imgtk = ImageTk.PhotoImage(img)

        self.video_label.imgtk =imgtk
        self.video_label.configure(image=imgtk)

        self.after(10, self._loop)

    def on_close(self):
        self.stop()
        self.destroy()

if __name__ == "__main__":
    App().mainloop()






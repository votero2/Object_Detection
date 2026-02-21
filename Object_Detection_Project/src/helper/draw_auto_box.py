import cv2
import numpy as np

def draw_auto_box(frame):
    #Prep
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7,7), 0)

    # Edges
    edges = cv2.Canny(blur, 50,150)
    edges = cv2.dilate(edges, None, iterations=2)
    edges = cv2.erode(edges, None, iterations=1)

    # Contours
    contours, _ = cv2.findContours(edges,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return frame, None

    # Biggesr contour -> Box
    c = max(contours, key= cv2.contourArea)
    if cv2.contourArea(c) < 1500:
        return frame, None

    x, y , w, h = cv2.boundingRect(c)

    #Draw
    cv2.rectangle(frame, (x,y), (x + w, y + h), (0,255,0),2)
    return frame, (x,y,w,h)

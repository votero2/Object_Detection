import cv2

class Camera:
    def __init__(self, index = None,width=640,height =480, max_indexes = 100):
        """
        index=None -> auto-detect camera(prefered)
        index=int -> force a specific camera index
        """
        self.index = index if index is not None else self._find_camera(max_indexes)
        if  self.index is None:
            raise RuntimeError("No camera device found")

        self.cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)

        if not self.cap.isOpened():
            raise RuntimeError(f"Camera index {self.index} not available")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT,height)

        print(f"[Camera] Using camera index{self.index}")

    
    def _find_camera(self, max_indexes):
        """
        Try camera indexes and return the first one that opens.
        Usually:
        - 0 = laptop webcam
        - 1 or 2 = Iriun (phone)
        """
        for i in range(1,max_indexes):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                cap.release()
                return i
            cap.release()

        cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
        if cap.isOpened():
            cap.release()
            return 0
        return None

    def read(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame


    def release(self):
        self.cap.release()





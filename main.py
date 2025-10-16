
import cv2 as cv
import numpy as np

width, height = 320, 320
dst_points = np.array([[0,0],[width,0],[width,height],[0,height]], dtype=np.float32)

cap = cv.VideoCapture(2)
cv.namedWindow("Webcam")
qr_detector = cv.QRCodeDetector()
homography = None

while True:
    ret, frame = cap.read()
    if not ret: break

    display_frame = frame.copy()
    data, pts, _ = qr_detector.detectAndDecode(display_frame)

    if pts is not None:
        pts = pts[0].astype(int)
        cv.polylines(display_frame, [pts], isClosed=True, color=(0,255,0), thickness=2)

    cv.putText(display_frame, "Presione C para capturar el QR", (10, 30), 
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    cv.imshow("Webcam", display_frame)

    key = cv.waitKey(1) & 0xFF

    if key == ord('c') and pts is not None:
        pts_float = pts.astype(np.float32)
        homography = cv.getPerspectiveTransform(pts_float, dst_points)
        frontal = cv.warpPerspective(frame, homography, (width, height))
        cv.imshow("Frontal", frontal)

    elif key == ord('d'): cv.destroyWindow("Frontal")

    elif key == ord('q'): break

cap.release()
cv.destroyAllWindows()

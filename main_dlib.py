import matplotlib.pyplot as plt
import dlib
from imutils import face_utils
import cv2

frame = cv2.imread('misc-faces.jpg')
grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


face_detect = dlib.get_frontal_face_detector()

rects = face_detect(grayscaled_img, 1)

count = 0

for (i, rect) in enumerate(rects):
    (x,y,w,h) = face_utils.rect_to_bb(rect)
    cv2.rectangle(grayscaled_img, (x,y), (x+w, y+h), (255, 255, 255), 3)
    count+=1

plt.figure(figsize=(8,8))
plt.imshow(grayscaled_img, cmap='gray')
plt.show()

print(count)

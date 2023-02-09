import matplotlib.pyplot as plt
import dlib
import cv2

frame = cv2.imread('misc-faces.jpg')
grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


face_detector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")

rects = face_detector(grayscaled_img, 1)

counts = 0
for (i, rect) in enumerate(rects):
    x1 = rect.rect.left()
    y1 = rect.rect.top()
    x2 = rect.rect.right()
    y2 = rect.rect.bottom()

    cv2.rectangle(grayscaled_img, (x1, y1), (x2,y2), (255, 255, 255), 3)
    counts+=1

plt.figure(figsize=(8,8))
plt.imshow(grayscaled_img, cmap="gray")
plt.show()
print(counts)
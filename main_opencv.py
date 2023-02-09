import cv2
import matplotlib.pyplot as plt

faceCascade = cv2.CascadeClassifier("data/haarcascade_frontalface_default.xml")
eyeCascade = cv2.CascadeClassifier("data/haarcascade_eye.xml")
smileCascade = cv2.CascadeClassifier("data/haarcascade_smile.xml")

# # Load image
frame = cv2.imread('misc-faces.jpg')
grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#Detecting faces
faces = faceCascade.detectMultiScale(grayscaled_img, scaleFactor=1.1, 
                                        minNeighbors=5, flags=cv2.CASCADE_SCALE_IMAGE)

count = 0
# For each face
for (x, y, w, h) in faces:
    #Draw rectangle around the face
    cv2.rectangle(grayscaled_img, (x,y), (x+w, y+h), (255, 0, 255), 3)
    count+=1

plt.figure(figsize=(8,8))
plt.imshow(grayscaled_img, cmap='gray')
plt.show()

print(count)


import cv2
import matplotlib.pyplot as plt


# Create tracker object
cap = cv2.VideoCapture("Videos/carsVideo.mp4")

currentframe = 0
while currentframe < 2 :
    ret,frame = cap.read()
    if ret:
        # if video is still left continue creating images
        name = 'Masks/' + str(currentframe) + '.jpg'
        print ('Creating...' + name)
        # writing the extracted images
        cv2.imwrite(name, frame)
        # increasing counter so that it will
        # show how many frames are created
        currentframe += 1
    else:
        break

img = cv2.imread('Masks/1.jpg')

plt.imshow(img)
plt.show()
cv2.waitKey(0)


# python yolo-mask.py
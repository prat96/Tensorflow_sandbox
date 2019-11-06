import os
import cv2
import numpy as np


def bpcr(image):
    y = [229]
    x = [329]
    index = 0
    while (index < len(x)):
        print("initial =", image[y[index], x[index]])
        image[y[index], x[index]] = int((image[y[index] - 1, x[index]] + image[y[index] + 1, x[index]] + image[y[index], x[index]] - 1 +
                     image[y[index], x[index] + 1]) / 4)
        print("after =", image[y[index], x[index]])
        index = index + 1
    return image


def read_video():
    cap = cv2.VideoCapture('thermal_wp2.mp4')
    if (cap.isOpened() == False):
        print("Error opening video stream or file")
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    out = cv2.VideoWriter('outpy.mp4', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (frame_width, frame_height))

    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imshow('Original', frame)

            # do bpcr here
            image = np.array(frame)
            corr_image = bpcr(image)

            out.write(corr_image)
            cv2.imshow('Corrected', corr_image)

            if cv2.waitKey(30) & 0xFF == ord('q'):
                break

        else:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # bpcr(image)
    read_video()

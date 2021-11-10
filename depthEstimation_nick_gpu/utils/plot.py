import cv2

def showInMovedWindow(winname, y, x):
    cv2.namedWindow(winname)       # Create a named window
    cv2.moveWindow(winname, y, x)  # Move it to (y, x)

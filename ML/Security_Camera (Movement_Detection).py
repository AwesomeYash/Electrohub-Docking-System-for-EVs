# Security Camera ( Movement Detection !!! )

# import the opencv library
import cv2
import winsound 
  
# define a video capture object
cam = cv2.VideoCapture(0)

"""  
while(True):
      
    # Capture the video frame
    # by frame
    ret, frame = cam.read()
  
    # Display the resulting frame
    cv2.imshow('frame', frame)
      
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
cam.release()
# Destroy all the windows
#cv2.destroyAllWindows()
"""

while cam.isOpened():

    #To make green rectangle that detect movements
    ret, frame1 = cam.read()
    ret, frame2 = cam.read()
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dialated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(dialated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    for c in contours:
        if cv2.contourArea(c) < 5000 :
            continue

        w, x ,y, h = cv2.boundingRect(c)
        cv2.rectangle(frame1, (x,y), (x+w, y+h), (0,0,255), 2)
        winsound.PlaySound('alert.wav', winsound.SND_ASYNC)
    
    if cv2.waitKey(10)== ord('q'):
        break

    cv2.imshow('Security Camera', frame1)
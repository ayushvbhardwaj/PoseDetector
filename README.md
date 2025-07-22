SetUp

**pip install opencv-python mediapipe**

------------------------------------------

preferable in a separate env in python 3.9:

Conda command to create new enviornment:

**conda create -n opencv_env python=3.9 -y**   

---------------------------------------------

**Poses**

1. Takeoff: both arms up
2. Land: arms crossed
3. Hover: both arms down resting
4. Left: one arm horizontally left
5. Right: one arm horizontally right
6. Forward: both arms pointing forward



---------------------------------------------

#Aruco Detector

**detects aruco markers through the webcam** 

Right now detect 6x6 markers can change in code if needed to do 4x4



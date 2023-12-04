# machine-vision
**Group**: Gesture Geniuses

**Names**: Taha Kazi, Saad Afridi, Daria Illianova, Carlos Saputra

**Problem Statement**: Accurate and real-time estimation of 3D hand pose from single depth images is a pressing challenge in computer vision. Existing methods face limitations in precision and robustness, particularly in handling joint ambiguities, and complex hand poses. This project aims to utilize Machine Learning to improve the accuracy and reliability of 3D hand pose estimation, enhancing its applicability in virtual reality, gesture recognition, and human-computer interaction systems.
![image](https://github.com/CSC490-Capstone-Design-Course/machine-vision/assets/47696403/e73f691d-0e40-4867-ac48-723909b30d0b)

**Dataset**: We are working with the ICVL dataset in our project. The dataset is 2GB in size – not as large as our previous chosen dataset, which is good for efficiency in processing and training. It contains a total of 5000 images with a 240x320 resolution for each image.

Furthermore,there is labelled data available, where each image has 16 points in total, corresponding to the 16 joint locations, in their 3D (x,y,z) position. [1]

The order of 16 joints is Palm, Thumb root, Thumb mid, Thumb tip, Index root, Index mid, Index tip, Middle root, Middle mid, Middle tip, Ring root, Ring mid, Ring tip, Pinky root, Pinky mid, Pinky tip. [1]

![image](https://github.com/CSC490-Capstone-Design-Course/machine-vision/assets/47696403/ba6f34b1-6181-40a9-9b23-8246f4dd40af)

 https://labicvl.github.io/hand.html 

**Goal**: We want to estimate 3D pose of articulated human hands using single depth images only. 

**TODO**: We still need to work on Model Architecture & Design, Loss Function Design, Training Strategy and Evaluation Metrics.


# 3D Handpose Estimation using Single Depth Images (CSC490H5F)
**Group**: Gesture Geniuses

**Group Members**: Taha Kazi, Saad Afridi, Daria Illarionova, Carlos Saputra

**Problem Statement**: Accurate and real-time estimation of 3D hand pose from single depth images is a pressing challenge in computer vision. Existing methods face limitations in precision and robustness, particularly in handling joint ambiguities, and complex hand poses. This project aims to utilize Machine Learning to improve the accuracy and reliability of 3D hand pose estimation, enhancing its applicability in virtual reality, gesture recognition, and human-computer interaction systems.

 [**Dataset**](https://labicvl.github.io/hand.html): We are working with the ICVL dataset in our project. The dataset is 2GB in size – not as large as our previous chosen dataset, which is good for efficiency in processing and training. It contains a total of 5000 images with a 240 x 320 resolution for each image. Furthermore, there is labelled data available, where each image has 16 points in total. Each line of labelled data is corresponding to each hand, which is represented by the 16 joint locations in their 3D (x,y,z) position. [1] The order of 16 joints is Palm, Thumb root, Thumb mid, Thumb tip, Index root, Index mid, Index tip, Middle root, Middle mid, Middle tip, Ring root, Ring mid, Ring tip, Pinky root, Pinky mid, Pinky tip. [1] 

 An image labelling each of the hand joints that are tracked [2]:
 
![image](https://github.com/CSC490-Capstone-Design-Course/machine-vision/assets/47696403/2e08181d-50d8-4555-9220-2b09c8ddad96)

A sample of hand images is available below, to get a visual sense of the ICVL dataset we used [3]:

![image](https://github.com/CSC490-Capstone-Design-Course/machine-vision/assets/47696403/a3e62f5b-0e5e-4448-99ed-f38958216a44)

In terms of input it is a numpy array of floats from 0-1, of dimensions of 240 x 320. While the output are 48 coordinate predictions in (x,y,z) format for each joint.

Example (to give an idea of input and output):

_Input:_
![image](https://github.com/CSC490-Capstone-Design-Course/machine-vision/assets/47696403/80bcc8cb-85fc-4de9-ab7c-0eaae46f14b0)

_Output:_
[229.35170122234697, 176.93892632204103, 346.0938500204286, 234.09473436484353, 149.79878599343675, 171.91208128766164, 260.76329734854556, 85.6191004596873, 220.31255828441894, 328.7552594572054, 267.5338875707244, 96.25667124558079, 71.75225028907839, 228.68347289130736, 176.44371319298077, 218.54134057844624, 288.57488379261156, 11.976376178444115, 188.90143947851473, 42.06807276008369, 288.42814334961116, 345.81781319295095, 84.65350891351662, 139.87000562883227, 343.1661787351584, 182.7833683450211, 333.9513382997157, 361.883955441314, 109.02769366275282, 68.85865279825802, 17.908231144229035, 182.1654524933387, 181.16114377671738, 58.34662287451815, 220.96232693136355, 36.092484827209546, 383.4765393150036, 176.4552414706205, 186.71472676776736, 256.69338129081973, 75.48744706152344, 178.9524916458292, 199.42034906080767, 8.411846451630334, 179.124378067366, 393.6306223961567, 385.62926715913886]


 **Implementation Overview**: 

**Evaluation Results**: In terms of our Evaluation results we looked at 3 major values, namely Mean Absolute Error value (MAE), Median Absolute Error value, and Standard Deviation of Residuals value:

1. Mean Absolute Error (MAE) value: 11.9, suggesting a moderate level of error in the predictions.
2. Median Absolute Error value: 9.9, suggesting that ¼ of the model predictions have an absolute error of less than 9.9 units.
3. Standard Deviation of Residuals value: 15.3, suggesting that there is a slight variability in the model’s predictions. Ultimately highlighting that certain handposes or scenarios yield more accurate predictions, while others may result in larger errors.

Below are the results from the console:

Evaluate the Test Dataset
84/84 [==============================] - 777s 9s/step - loss: 11.9997 - mae: 11.9997
test loss, test acc: [11.999682426452637, 11.999682426452637]
Finished Evaluating the Test Dataset

**Contributions**:

Saad Afrdi:

Taha Kazi:

Carlos Saputra:

Daria Illarionova:

**References**:

[1] Imperical College of London. (n.d.). 3D articulated hand pose estimation with single depth images. 3D Hand Pose Estimation. https://labicvl.github.io/hand.html 

[2] N. Otberdout, L. Ballihi and D. Aboutajdine, ”Hand pose estimation based on deep learning depth map for hand gesture recognition,” 2017 Intelligent Systems and Computer Vision (ISCV), Fez, Morocco, 2017, pp. 1-8, doi: 10.1109/ISACV.2017.8054904.

[3] Tang, D., Chang, H. J., Tejani, A., &amp; Kim, T.-K. (2017b). Latent regression forest: Structured estimation of 3D Hand poses. IEEE Transactions on Pattern Analysis and Machine Intelligence, 39(7), 1374–1387. https://doi.org/10.1109/tpami.2016.2599170 


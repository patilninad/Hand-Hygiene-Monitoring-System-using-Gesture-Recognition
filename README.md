# Hand-Hygiene-Monitoring-System-using-Gesture-Recognition 
This monitoring system helps to check the compliane of proper hand hygiene.Practicing hand hygiene is a simple yet effective way to prevent infections. Hospital staff tends to perform hand hygiene properly often only when they are aware that they are being observed. To track compliance, we aim to design a simple, cost‐effective solution easily reproducible on a mass scale which will in real time, monitor the entire process of hand washing.  
There are 8 steps recommended by WHO to properly maintain hand hygiene.These 8 steps are described below:  
![](https://github.com/patilninad/Hand-Hygiene-Monitoring-System-using-Gesture-Recognition/blob/master/washing_hands_photos.jpg)  
In this project, we present a hand gesture monitoring and reminder system that monitors hand hygiene by checking each and every step using gesture recognition and provides real time feedback,if any step is not done properly. Aim is to develop a working product that can differentiate different hand gestures with a high accuracy.Product must be robust, scalable, and easy to install.   
Image processing along with deep learning is used to train a neural network model on the 8 steps.  
The current epidemic of the novel Covid-19 makes it essential for everyone to wash their hands properly as shown in the image above. 
Currently the system uses MobileNetV2 on tensorflow back-end for keras. It manages to classify the 8 different steps with close to 70% accuracy.  
We are thinking about how we can implement a 20 second timer.The system will monitor whether the person has performed all the 8 steps for a total duration of 20 seconds as recommended by WHO.  
We are also using NVIDIA DIGITS web based API as an alternative approach for training along with the standard way of training.  
I had previously worked on NVIDIA DIGITS for a different task based on ConvNet training.  
# Here is the GITHUB link of my project:  
How to install NVIDIA DIGITS:  
https://github.com/patilninad/DIGITS  

Project:  
https://github.com/patilninad/Training

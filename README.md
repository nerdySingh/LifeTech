# Deep Learning Powered Arrhythmia (Heart Disease) Detector

project page at https://devpost.com/software/lifetech

## Sample UI -
![alt text](https://github.com/ironhide23586/Life-Tech/blob/master/HeartbeatML/gallery/UI.jpg)
 
## Hardware sample snap -
 ![alt text](https://github.com/ironhide23586/Life-Tech/blob/master/HeartbeatML/gallery/hardware.jpg)
 
## Performance statistics -
 ![alt text](https://github.com/ironhide23586/Life-Tech/blob/master/HeartbeatML/gallery/stats.jpg)

The ESP 8266 is reading the data from the heartbeat sensor and is connected to wifi. It then pushes this data to the Amazon Web Service which has a RDS instance up and running with a MySql already setup on it. We store the data on this mysql database and we use flask to publish this data on the server that runs an ubuntu image. 

The data is picked up by the Deep Learning Inference Engine trained on the MIT arrhythmia dataset to identify the bad heartbeats from the good ones.

The data is picked up in a JSON format where the heart rate monitor UI will show user’s heartbeat being generated in real time. We have a software trigger that will tell us whether the user is suffering from a heart attack and using Twilio application, it will send SMS to user’s family and doctor for urgent care. User will also get one voice message which consists details of what are the precautions user can take for heart attack before any emergency care arrives.

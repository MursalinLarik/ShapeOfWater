# Shape Of Water
Shape of Water is a civic innovation project aimed at resolving the water issues in the Megacity Karachi. We do so by first identifying the lack of existing maps of the water supply network. GPR (Ground Penetrating Radar) is a device which captures B-scan radargrams of the underground 2D area. At the core of the project, we try to classify GPR radargrams at first then detect the pipes in the radargrams and then calculate their depth in subsurface. This information coupled with the embedded GPS location is visualized on a geographical map using QGIS. This CNNs, GAN models and digital image processing algorithms were used in the process to reach the end goal. 
<br />
<br />
First of all using the handcrafted digital image processing algorithm on the data collected using the GPR on a pilot area where there are pipes present underground, following results were obtained:
![alt text](https://github.com/MursalinLarik/ShapeOfWater/blob/master/object_detection_image_processing.png)
<br />
<br />
In the deep learninig approach, we have used several algorithms and techniques. The classification is peformed using both, MobileNet and Inception Net. Then, for the detection of the hyberbola formed due to the dispersion of EM waves by GPR over the pipes (signifying the presence of pipes), we have used Faster RCNN. Furthermore, GANs are also used to simulated some of the data, however the results from GANs are not used towards the end product of the map.
<br />
<br />
The accuracy and the loss from the classification models are recorded. The results from MobileNet are below:
![alt text](https://github.com/MursalinLarik/ShapeOfWater/blob/master/classification_accuracy.png)
![alt text](https://github.com/MursalinLarik/ShapeOfWater/blob/master/classification_loss.png)
<br />
<br />
The results from the object detection algorithm of Faster RCNN yielded the following result:
![alt text](https://github.com/MursalinLarik/ShapeOfWater/blob/master/faster_rcnn_object_detection.png)
<br />
<br />
Using GANs, we were able to generate some results similar to the actual B-Scan radargrams:
![alt text](https://github.com/MursalinLarik/ShapeOfWater/blob/master/GAN_architecture.jpg)
<br/>
![alt text](https://github.com/MursalinLarik/ShapeOfWater/blob/master/Gan_results.png)
<br />
<br />
Finally the results coupled with the GPS coordinates of the collected dataset were plotted on the ma using QGIS software:
<br />
![alt text](https://github.com/MursalinLarik/ShapeOfWater/blob/master/plotting_on_map.png)

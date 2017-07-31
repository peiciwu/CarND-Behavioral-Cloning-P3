# **Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./writeup_image/NVIDIA_model.png "NVIDIA architecture"
[image2]: ./writeup_image/center.jpg "Original image (before cropping)"
[image3]: ./writeup_image/centerCropped.jpg "Cropped image (after cropping)"
[image4]: ./writeup_image/bridge.jpg "On the bridge"
[image5]: ./history.png "Loss trend"

## Files Submitted

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results
* video.mp4 records of driving the car in autonomous mode for more than one lap. (FPS is set to 48.)

## Model Architecture and Training Strategy

### My first implementation

#### 1. Model architecture

I first implemented a model simply based on the one shown in Chapter 14 in the class materials. The model essentially is based on the NVIDIA architecture. First is a normalization that is to use a Keras lambda layer, and followed by three convolution layers, each with a 5x5 kernel, 2x2 striding, and output depths are 24, 36, and 48, respectively. Next, there are two more convolution layers, both with a 3x3 kernel and output depth of 64. Then, there are three fully connected layers, where the output sizes are 100, 50, 10, respectively. Activation on each layer is using RELU layers that help introduce nonlinearity. Adam is used as the optimizer and using mean square loss.

In addition, there is a cropping layer after the normalization layer. This cropping layer is needed as our data contains some unnecessary information, e.g. sky, trees, rocks, and part of the car itself. Thus, the layer is to crop 70 pixels from the top and 25 pixels from the bottom of the image. Example images of before and after the cropping are shown below. The impact of with and without the cropping layer would be discussed later. 

![alt_text][image2]
![alt_text][image3]

I've also implemented fit_generator for better efficiency, and the input data set is shuffled and split into a training set (80%) and validation set (20%). Other than the images of the center camera, images of left and right camera are used as well: steering angle is added 0.25 for the left camera images and -0.25 for the right camera images.

#### 2. Results of the first attempt

My initial training uses the data set provided by the class. Learning rate is set to 0.001, batch size is 32, and EPOCH is 10. By doing so, on EPOCH 10, the loss on the testing set is ~0.01 while the loss on the validation set is ~0.014. Unfortunately, the trained model cannot successfully drive around. The car either drove out of the road or got stuck on the bridge. After playing around with this data set, I decided to collect my own data set.

NOTE: in due course, I also found that the provided "drive.py" uses PIL.Image to read/write image, while my own "model.py" uses cv2 to read image. The difference here is PIL.Image stores a jpeg in RGB format, but cv2 stores in BGR format. Since my model directly works on the color image, this makes difference, thus, I've modified the provided "drive.py" to use cv2 to read/write image. This change does improve the autonomous driving results. Even my first model mentioned above can get some decent results.

### Strategy of collecting results

I recorded a bit more than two laps data, and paid a special attention on the section of entering and exiting the bridge. The reason is because the steering angle is normally 0 on the bridge, any angle adjustment on the bridge tends to make the car stuck on the bridge, which is the largest problem on my model (even on the improved model that will be mentioned later.) Thus, the goal is to collect data entering/exiting the bridge where the car is able to drive on the center of the road on the bridge. Below is an example of driving on the bridge.

![alt_text][image4]

#### Results on my own data

Parameter setting is same as the one mentioned above, other than EPOCH is increased from 10 to 15 (in order to get similar validation loss value.) With my own data set, on EPOCH 10, test loss is 0.0129 and validation loss is 0.0163. On EPOCH 15, test loss is 0.056 and validation loss is 0.0132. Despite of the large gap between testing loss and validation loss, the good thing is that the driving result becomes fairly acceptable. The car didn't hit any white/yellow lines and successfully drove one lap. The only nitpick is that the car drives very close to the road boundaries.

### Further improving

A dropout layer (with 20% dropout rate) is added after each fully connected layer in the hope that the gap between the testing loss and the validation loss can be reduced. While the gap does get decreased, the driving result doesn't make large difference, which led a conclusion that more data is needed in order to generalize the model.

I tried to collect some recover data, but it is not trivial to record on and off to collect meaningful data. So I gave up the idea of collecting more data. Instead, I decided to horizontally flip all my current images (and negative the steering angle), which can double the data set. With this new data set, on EPOCH 10, the training loss is 0.0172 and the validation loss is 0.0168. Below is the trend of loss on both training and validation set.

![alt_text][image5]

The model achieves very good driving results. The car drives mostly on the center of the road, and didn't frequently make small turns which makes a more smooth driving.

#### Parameters tuning and experiments

* Played with different values of batch size, but didn't have any decisive conclusion so stick to 32.
* Increased the learning rate got much worse loss value.
* Without the cropping layer, the testing and validation loss can become very small (e.g. 0.0009), however, this small loss doesn't reflect on the driving results, which makes sense, as if the image holds more information, more data is needed in order to generalize the model. Thus, in this case, removing unnecessary pixes is important with regards to generalizing the model.
* Instead of flipping all images, some criteria is added such to only flip an image if its steering angle is larger than 0.5 or less than -0.5. Reasoning here is to only flip images on sharp turns because they are more difficult to predict. However, in the end, flipping all got much better driving results. I would guess this is because my own data set is not large enough (the # of images is 14607 without any flipping), so more flipping, more data, and thus better results.
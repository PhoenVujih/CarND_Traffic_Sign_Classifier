# **Traffic Sign Recognition**


# Overview

In this project, I used deep neural networks and convolutional neural networks to classify traffic signs. I trained and validated two models so they could classify traffic sign images using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). After the model was trained, I then tried out my model on images of German traffic signs that I found on the web.

## Requirements
* Python
* Tensorflow
* Scikit Learn
* Pillow


[//]: # (Image References)

[image1]: ./figures/data_summary.png "Visualization"
[image2]: ./figures/sample_image.png
[image2_1]: ./figures/sample_image_gray.png "Grayscaling"
[image_new]: ./figures/new_images.png "new images"
[conv1_1]: ./figures/image_conv1_1.png "conv1_1"
[conv3]: ./figures/image_conv3.png "conv3"
[image4]: ./figures/image4.png "image4"

---

### Data Set Summary & Exploration

#### 1. Basic summary of the data set.

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. The exploratory visualization of the dataset.

The exploratory visualization of the data set.
The sample numbers of different classes are quite different which reflect the real number distribution of the traffic signs on the road.

![alt text][image1]   

### Model Architecture

#### 1. Data Preprocessing

As a first step, I decided to convert the images to grayscale. Unlike the traffic light, the sub-features of traffic signs are mostly hidden in the geometries of the traffic signs rather than the color.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]   ![alt text][image2_1]

After that, I processed the data using: ``(data-average_value)/standard_deviation``  to centralize and normalize the variance of the individual image.


#### 2. Model Architecture

##### LeNet

|Layer                       | Output Shape    |
|----------------------------|:--------:|
|Input                       | 32x32x1  |
|Convolution (valid, 5x5, stride 1, ReLU)    | 28x28x6  |
|Max Pooling (valid, 2x2, stride 2)    | 14x14x6  |
|Convolution (valid, 5x5, stride 1, ReLU)    | 10x10x6  |
|Max Pooling (valid, 2x2, stride 2)    | 5x5x6  |
|Convolution (valid, 5x5, stride 1, ReLU)    | 1x1x120  |
|Dense (ReLU)                       | 84      |
|Dropout (0.3)                      | 84    |
|Dense                       | 43       |
|Softmax       | 43       |

##### My Net
|Layer                       | Output Shape    |
|----------------------------|:--------:|
|Input                       | 32x32x1  |
|Convolution (same, 1x1, stride 1, ReLU)    | 32x32x32  |
|Convolution (same, 3x3, stride 1, ReLU)    | 32x32x32  |
|Convolution (same, 3x3, stride 1, ReLU)    | 32x32x32  |
|Convolution (same, 3x3, stride 1, ReLU)    | 32x32x32  |
|Max Pooling (valid, 2x2, stride 2)    | 16x16x32  |
|Convolution (same, 3x3, stride 1, ReLU)    | 16x16x32  |
|Convolution (same, 3x3, stride 1, ReLU)    | 16x16x32  |
|Convolution (same, 3x3, stride 1, ReLU)    | 16x16x32  |
|Max Pooling (valid, 2x2, stride 2)    | 8x8x32  |
|Convolution (same, 3x3, stride 1, ReLU)    | 8x8x32  |
|Convolution (same, 3x3, stride 1, ReLU)    | 8x8x32  |
|Convolution (same, 3x3, stride 1, ReLU)    | 8x8x32  |
|Max Pooling (valid, 2x2, stride 2)    | 4x4x32  |
|Convolution (valid, 4x4, stride 1, ReLU)    | 1x1x512  |
|Dropout (0.3)                       | 1x1x512      |
|Dense (ReLU)                       | 120      |
|Dropout (0.3)                       | 120     |
|Dense                        | 43      |
|Softmax       | 43       |


#### 3. Model Training

To train the model, I used an Adam Optimizer with the learning rate 0.001.
I ran 200 epochs to train My Net for the first round and then ran another round to pick the best result.
For the LeNet, I ran 200 epochs just to see the performance of the model.
The batch size was set 500 for both models.
To minimize the influence of overfit and improve the performance of the model, dropout was utilized before the dense layer with the keep probability 0.3.


#### 4. Discussions

The first model I tried is LeNet since it performs well in letter recognition.
The original version of LeNet has no dropout layer and I got the test accuracy around 0.91. After I applied dropout layer before the dense layer, the test accuracy increased to 0.93.

The idea of MyModel was inspired by VGG net. Although the window size of each convolution layer is only 3x3, with the increase of layer number, the effective receptive field and the non-linearity also increase which make the model has potentially better performance of classification.

Since the test accuracy of the model fluctuated during the training, I added a couple of lines of code to record the best test accuracy so far and saw if the model after the next epoch performed better or not. If better, the save the checkpoint of the model.

Finally, I got a trained model that performed well on traffic sign classification.

My final model results were:
* training set accuracy of 1.00
* validation set accuracy of 0.988
* test set accuracy of 0.977



### Test a Model on New Images

#### 1. Resize the new images

I downloaded five new images from the internet and resized them to make sure they are 32x32.

![alt text][image_new]


#### 2. Results of the prediction

Here are the results of the prediction:

| Image			        |     Prediction	       			|
|:---------------------:|:-----------------------------:|
| Go straight or right		| Go straight or right				|
| Stop Sign      		| Stop Sign		                	|
| Speed limit (30km/h) 		| Speed limit (30km/h)				|
| Right-of-way at the next intersection	| Right-of-way at the next intersection	|
| Road work    			| Road work					|

The model successfully identify the all 5 traffic signs, which gives an accuracy of 100%.
Consider the number of the samples is limited (only 5), 100% accuracy cannot tell that the model is absolutely accurate. But it truly did well on the given dataset.

#### 3. Top 5 softmax probabilities for each image (cell 17 in the notebook)


The top five soft max probabilities were as follows

| Image     	|     Class   |1st guess|2nd guess|3rd guess|4th guess|5th guess|
|:------------:|:---------:|:--------:|:--------:|:--------:|:--------:|:------:|
| 0 | Go straight or right			|Go straight or right: 0.99998|Turn left ahead: 1.96E-05| No entry: 1.13E-06| Children crossing: 3.54E-07| Road narrows on the right: 2.38E-07|
| 1 | Stop Sign			|Stop Sign: 1.00| Speed limit (120km/h): 7.42E-09| Turn left ahead: 2.89E-09| Yield: 2.22E-09| Speed limit (30km/h): 1.22E-09|
| 2	| Speed limit (30km/h)	|Speed limit (30km/h):0.99973 		| Keep left: 1.12E-4|Speed limit (20km/h): 5.15E-5|Keep right: 4.99E-5|Speed limit (50km/h): 3.30E-5|
| 3	| Right-of-way at the next intersection		|Right-of-way at the next intersection: 1.00| Wild animals crossing: 1.38E-28| No entry: 5.11E-29| General caution: 7.92E-30| No passing for vehicles over 3.5 metric tons: 3.03E-30	|
| 4	| Road work	|Road work: 1.00|Keep right: 1.41E-22|Road narrows on the right: 1.96E-24|Wild animals crossing: 1.82E-24|General caution: 4.85E-25|

It seems that the model is pretty sure for the prediction and softmax probability of the top guess is almost 1.0.



### Visualizing the Neural Network

The input image is as follows.

![alt text][image4]

The figure below shows the visualization of the data after the second convolution layer.

The 32 processed figures show that some simple features were detected in the image such as the edges at different directions. The circle and two arrows are clearly identified in the image. Some of the images are totally black, which may attribute to the deactivation.

![alt text][conv1_1]

While after several layers, the processed data seems more abstract as shown below. The figure shows the result after the last second convolution layer. Only some dots in the figures which represent the high level features of the original picture. 

![alt text][conv3]

# **Traffic Sign Recognition**
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)




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

### Model Performance
* training set accuracy of 100.0%
* validation set accuracy of 98.8%
* test set accuracy of 97.7%

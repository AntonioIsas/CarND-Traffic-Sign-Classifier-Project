# **Traffic Sign Recognition**
---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/dataSet_dist.png "Visualization"
[imgSample]: ./examples/sample_images.png "Sample Images"
[preprocessed]: ./examples/preprocessed.png "Sample Images"
[newImages]: ./examples/new_images.png "New Images"

[yield]: ./examples/yieldSign.png "Yield"
[featureMap]: ./examples/featureMap.png "Feature Map"

## Rubric Points
---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/AntonioIsas/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is **34799**
* The size of the validation set is **4410**
* The size of test set is **12630**
* The shape of a traffic sign image is **(32, 32, 3)**
* The number of unique classes/labels in the data set is **43**

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed among the labels

![alt text][image1]

And a sample of some of the images to see how they look

![alt text][imgSample]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As part of the preprocessing I tested with 3 different color spaces, normal, grayscale and hsv. At first I thought that the color one would give better result because it provides additional information in the form of color but looks like the nodes don't like it much as it was better with grayscale, I tried the hsv because it gives good results in computer vision so I thought the computer might look at something else but the test was giving an accuracy worse than the normal space, later I found that the convolutions were not being activated so this color space is not very good as too much information is lost   

Here is an example of a traffic sign image in the different color spaces.

![alt text][preprocessed]

As a last step, I normalized the image data to provide numerical stability and to help the optimizer find a solution faster

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		    | 32x32x1 grayscale image   							|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 32x32x64 	|
| RELU					        |												|
| dropout					      |												|
| Max pooling	      	  | 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5       | 1x1 stride, valid padding, outputs 10x10x16      									|
| RELU					        |												|
| dropout					      |												|
| Max pooling	      	  | 2x2 stride,  outputs 5x5x16 				|
|	Flatten				        |	outputs 400											|
| Fully connected		    | outputs 120       									|
| RELU					        |												|
| dropout					      |												|
| Fully connected		    | outputs 84       									|
| dropout					      |												|
| Fully connected		    | outputs 43      									|



#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used a batch size of 50, and 100 epochs, using this small batch size allows the model to increase accuracy faster instead of using many more epochs
learning rate of .001
and beta of .001

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of **99**
* validation set accuracy of **95**
* test set accuracy of **93**

I decided to test LeNet-5 as I had used it previously and it had good results classifying images.
After the initial test it had a good accuracy so I choose to keep this model and just tune it as needed, after preprocessing the images and tunning the parameters I noticed the training accuracy at 100 while the validation was below 93, meaning the model was over fitting so I added regularization and dropout this allows the accuracy to go up to 95

The validation and test accuracy have a good percentage indicating that the model is working well

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are the new images, I grabbed this from a video of someone driving through Germany

![alt text][newImages]

The first 5 images were too clear and the model always classified them correctly so I grabbed another one which is harder because its darker smaller and is a little blurry, in this case it also classifies correctly but its softmax probability is not as high as in the other cases

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| General caution     		| General caution									|
| No entry 			| No entry								  |
| Ahead only		| Ahead only   							|
| Keep right		| Keep right  							|
| 120 km/h	    | 120 km/h					 				|
| 80 km/h	      | 80 km/h					 				  |
| Yield					| Yield											|
| Yield					| Yield											|
| 30 km/h	      | 30 km/h					 				  |
| Road work			| Road work										|

The model was able to correctly guess 10 of the 10 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 93%
Initially I had only 5 images but wanted to test on some harder ones so I grabbed 5 more

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 16th cell of the Ipython notebook.

For all the images that I found the model was very confident of the answer with the smallest one being the 9th image with only 84, all others where above 93

1. *General caution*

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .99993       			    | General caution  									|
| .00006  				      | Pedestrians 										|
| .00001			          | Traffic signals									|

2. *No entry*

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1.00000      			    | No entry   									|

3. *Ahead only*

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1.00000      			    | Ahead only   									|

4. *Keep right*

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1.00000      			    | Keep right  									|

5. *Speed limit (120km/h)*
| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 0.99113      			    | Speed limit (120km/h)									|
| 0.00538    				    | Speed limit (100km/h)										|
| 0.00155					      | Speed limit (70km/h)										|
| 0.00081      			    | Speed limit (80km/h)					 				|
| 0.00077			          | Speed limit (20km/h)     							|

6. *Speed limit (80km/h)*
| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 0.93456     			    | Speed limit (80km/h)									|
| 0.02827   				    | Speed limit (50km/h)										|
| 0.01556					      | Speed limit (60km/h)										|
| 0.01229     			    | Speed limit (70km/h)					 				|
| 0.00331 		          | No passing for vehicles over 3.5 metric tons    							|

7. *Yield*

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1.00000      			    | Yield  									|

8. *Yield*

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 0.99445    			      | Yield								|
| 0.00312   				    | Go straight or right									|
| 0.00134   			      | Ahead only									|
| 0.00022     			    | Turn right ahead					 				|
| 0.00021 		          | Turn left ahead    							|

9. *Speed limit (30km/h)*

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 0.84942      			    | Speed limit (30km/h)									|
| 0.07271    				    | Speed limit (70km/h)										|
| 0.02658					      | Speed limit (20km/h)										|
| 0.02129      			    | Speed limit (50km/h)					 				|
| 0.00787			          | Speed limit (80km/h)     							|

10. *Road work*

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 0.99210      			    | Road work								|
| 0.00355    				    | Dangerous curve to the right										|
| 0.00138					      | General caution										|
| 0.00041     			    | Keep right				 				|
| 0.00038			          | Traffic signals     							|

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

For this yield image, the feature maps seems to be identifying the borders, there seems to be one feature map for each border and some of them are combined
![alt text][yield]

![alt text][featureMap]



[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4


### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

In the 6th cell underneath the section "Feature Extraction Functions" I define the get_hog_features() function. This takes in all of the parameters for the standard hog() function. The purpose of putting it in a separate function is to easily return the visualization image if specified. Otherwise it just returns the feature vector that was extracted by the hog() function. 

The 2nd cell underneath the section "CLASSIFIER" is where I actually called the get_hog_features() for each training image. I passed each list of image paths (cars_train and non_cars_train) through the extract_features() function which is the cell directly below the get_hog_features() function mentioned above. The extract_features() function takes in the list of image paths, reads in each image, and calls the get_hog_features() function, as well as any other specified feature extraction function to create a complete feature vector. This returns 2 lists of feature vectors (cars_features and non_cars_features) to be used for training.


#### 2. Explain how you settled on your final choice of HOG parameters.

The first cell in the section "CLASSIFIER" contains all of the parameters I used for feature extraction. The ones relevant for HOG are the first 4 (orient, pix_per_cell, cell_per_block, cspace) as well as well as the 7th and 8th (hog_channel and hog_feat). The hog_feat parameter was for use in the extract_features() function to specify whether I wanted to include HOG features at all in my feature vectors. After experimenting I ended up with the following parameters:

orient = 9
pix_per_cell = 8
cell_per_block = 2
cspace = 'YCrCb'
hog_channel = 'ALL'

Here are the results on one of the test images:
![Alt text](output_images/hog_features_car.png?raw=True)

And here is a non car image:
![Alt text](output_images/hog_features_noncar.png?raw=True)


I tried more orientations and fewer pixels per cell, but this resulted in a vector that was much too long for extracting and processing. I also experimented with a greater number of pixels per cell, but the accuracy decreased.


Along with HOG features, I also used spatial features and color histogram features. The parameters used for this can be found in the same cell as the hog parameters. All together I ended up with a total feature vector size of 8460. It took me about 11 minutes to extract all of the features for training. For this reason I ended up saving and reloading my trained classifer to avoid having to extract features for training each time. I then scaled the training and test data using a scaler that I fit on only the training data.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The cell under the section "BUILD CLASSIFIER" is where I defined and trained the classifier. I started out using an SVC() classifier and using a grid search to find optimal parameters. This method took much longer to train and didn't provide much performance benefits over using a LinearSVC(). 

I trained the classifier on 14208 images and tested on 3552 images. My classifier gave an accuracy of 99.4% on the test set which was plenty good to successfully track vehicles in the video.


### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I opted to use the subsampling approach instead of the sliding window approach. Although video processing still took very long, this sped things up substantially. This function is defined in the 11th cell under the "Feature Extraction Functions" section. 

I used 3 scales (1, 1.5, 2). This provided me with 3 window sizes of 64, 96, and 128 and a total number window count of 202 windows.

to capture different vehicle sizes. I implemented these windows at 3 different y locations in the images ([400,496],[400,528],[464,660]). This put the smaller windows nearer the horizon and the larger windows nearer the car to capture the different apparant vehicle sizes at those locations.

I also decided to restrict the x dimension ([300,1280]). For the purposes of this project it helped to ignore cars on the opposite side of the road. This is a less robust implemention and something I plan to change in the future.

Here is a picture of the window coverage for each scale:
scale = 1 
ybounds = [400,496]
![Alt text](output_images/small_windows.png?raw=True)

scale = 1.5
ybounds = [400,528]
![Alt text](output_images/med_windows.png?raw=True)

scale = 2
ybounds = [464,660]
![Alt text](output_images/large_windows.png?raw=True)

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

My final decision was to use all 3 suggested feature vectors (hog, spatial, histogram). I tried to omit spatial and histogram, but I could not get sufficiently good results without them. I also decided on the YCrCb colorspace as this also provided the best results. Here are some of the test images of my results:
![Alt text](output_images/test1.png?raw=True)
![Alt text](output_images/test2.png?raw=True)
![Alt text](output_images/test3.png?raw=True)
![Alt text](output_images/test4.png?raw=True)

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](output_images/output_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I used the heatmapping method to combine overlapping detections and thresholding to eliminate false positives. The three functions I used are the last 3 functions in the "Feature Extraction Functions" section. First I defined an empty heatmap and added heat from the positive detections. Then I thresholded the map to eliminate any "cool" areas that were likely to be false positives. Then I used the label() function to identify the hot sections that were likely to be vehicles. I used the output from this function as my final set of bounding boxes to draw on the video.

This method alone was very helpful in making detections, but I also decided to add up heat maps over several frames and threshold the final summed heatmap. This method allowed me to threshold at a higher rate and prevent any false positives that were too hot in one or two successive frames from passing over the threshold. I did this by defining a Maps class and using a deque to save the heatmaps for the last 10 frames. I then used a threshold of 7 on this total. This number allowed me to eliminate all false positives from the video.

Here are some of the heat maps from the test images:
![Alt text](output_images/heat1.png?raw=True)
![Alt text](output_images/heat2.png?raw=True)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

There were 2 things that primarily stood out to me that could use improvement. 

First, even while experimenting with different feature parameters, I was never quite able to detect the white car soon enough into it appearing in the video. I would like to have detected it around 4 or 5 seconds of the video, but it only detects it at around 7 seconds.

Second, since I restricted the search in the X dimension, it is not as robust of an implementation as it could be. I plan on experimenting with the full X dimension in the future, but for the purposes of this project it worked out.


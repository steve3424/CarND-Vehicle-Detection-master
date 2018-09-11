This writeup is an explanation of the steps taken to detect vehicles in a video stream. I used a tradition computer vision approach where the features used to classify the objects are manually chosen and tuned before training the classifier. This is opposed to a deep learning approach where a deep neural network is constructed and manually tuned, but the network itself automatically learns what features to use in order to classify the objects. Both methods use supervised classification to train a classifier.

The basic pipeline takes 5 steps:
1. Acquire labeled training data
2. Transform training data into features
3. Train classifier on transformed data
4. Separate each video frame into segments
5. Run each segment through trained classifier

The rest of this writeup will go through these steps in more detail and explain how I was able to implement them. My notebook containing all of the code is not set up in the same order as this pipeline, so references made to the code will jump around a bit.

-------------------------------------------------------
## STEP 1. Acquire labeled training data
The training data used for this project was provided by Udacity. The dataset is from the KITTI dataset. It consists of 17,760 total images. 8792 are of vehicles and 8968 are of non-vehicles. The vehicles contain vehicles of different color, shape, and orientation. The non-vehicles contain various images of road segments and other background features. Each image in the data set was a 64x64 pixel image. Labels were manually created by creating a vector of 1s for each car image and a vector of 0s for each non-car image. Here are some images of each of the datasets.

VEHICLE TRAINING IMAGES
![Alt text](large_training_set\vehicles\GTI_Far\image0126.png?raw=True)
![Alt text](large_training_set\vehicles\GTI_Left\image011.png?raw=True)
![Alt text](large_training_set\vehicles\KITTI_extracted\23.png?raw=True)

NON-VEHICLE TRAINING IMAGES
![Alt text](large_training_set\non-vehicles\GTI\image13.png?raw=True)
![Alt text](large_training_set\non-vehicles\GTI\image1510.png?raw=True)
![Alt text](large_training_set\non-vehicles\Extras\extra21.png?raw=True)
![Alt text](large_training_set\non-vehicles\Extras\extra119.png?raw=True)

------------------------------------------------------------------
## STEP 2. Transform training data into features
This step is the meat of the traditional approach. The features that are used to represent the images determine how well the classifier can train be trained and subsequently classify objects in the video frames. Technically this step is optional. A classifier can simply be trained on the training data as is. Picking features to transform the images serves to improve performance.

The first feature to decide on is the colorspace of each image. I experimented with RGB, HSV, HLS, YUV, and YCrCb among others. I found that using the YCrCb colorspace provided me with the best results.

**PROVIDE VISUALIZATIONS SHOWING WHY THIS WAS THE BEST**

**TEST ON NON TRANSFORMED IMAGES AND COMMENT ON THAT METHOD HERE

The three feature extraction techniques that were introduced in the lectures were spatial binning, color histograms, and hog features. Through my own experimentation, I found the performance was the best when combining all 3 methods.

Spatial binning simply consisted of resizing the image to a smaller size and then flattening the image into a vector using numpy.ravel(). By down-sizing the image we are able to retain all of the inherent spatial features while reducing the amount of features we are working with. Turning the image into a vector simply makes it more wieldy to combine with the other feature vectors.

Resizing works as long as the image isn't resized to a point where the image is unrecognizable. This would amount to losing the essential features of the image. The size I settled on was 32x32. This effectively reduced the amount of features while still maintaining the image. I could have possibly gotten slightly more performance gains if I kept the size as is, but they likely would have been negligible and come at the cost of increasing the processing time of my pipeline. Here is a quick visualization of the spatial binning:

IMAGE
IMAGE
IMAGE
IMAGE

PROMPT: Is the resizing really only to increase processing time by reducing vector size??
PROMPT: Explore in greater depth why this would help to classify vehicles.

The next feature extraction technique was creating a histogram of the color features. This separates out each color channel and creates a histogram of the values in each channel. The 3 histograms are then flattened into vectors, combined with each other, then later combined with the other features. This process helps create a signature of the image in the colorspace that is being used.

For instance, in RGB space this might be detrimental. Since cars obviously come in a variety of colors, the histograms of red and blue cars may be completely different and actually make them more difficult to classify together.
**Test RGB with different color cars and see results**

The final and perhaps most important feature to use were HOG features. This takes a gradient of the image and creates a histogram of different sections of the image based on gradient magnitude and direction. This step allows us to create a signature of some of the important structural information of each image. Here is an example of the HOG features of a vehicle and non-vehicle image:

VEHICLE HOG
![Alt text](output_images\hog_features_car.png?raw=True)

NON-VEHICLE HOG
![Alt text](output_images\hog_features_noncar.png?raw=True)

As can be seen from the images, cars and non-cars have pretty distinct HOG signatures help the classifier's performance.

------------------------------------------------------------------
## Step 3. Train classifier on transformed data
Once we have decided which features we are going to use, the next step is to transform all of the training image and use them to train the classifier. This step is done in the 2nd cell underneath the "Classifier" section of my notebook.

First I load all of the training images into the notebook. Then I run them through my extract_features() function which takes in the parameters of all the feature extraction functions and transforms each image into feature vectors. After this I create the labels and split the data into training and validation sets. The video that the code is run on can be considered the final testing set. I then scale the data using sklearn.preprocessing.StandardScaler() before training the classifier. Finally I train a LinearSVC on the scaled data and test on the validation set. Using the features and parameters explained earlier I was able to consistently get a 99% or better accuracy on the validation set. 


-------------------------------------------------------------------------
## Step 4. Separate each video frame into segments
Now we have a classifier that has been trained on vehicle and non-vehicle images. Our goal though is not to simply classify vehicles from non-vehicles. Our goal is to do this *within a video feed of the street*. Because of this, we can't just put the video feed of a street through the classifier. Since it has not been trained on this kind of data, it will not work as desired.

The solution to this problem is to separate each video frame into smaller sections and run each of those sections through the classifier. Remember the classifier has been trained on 64x64 images containing just cars or non-car background. The size of the sections must break the image up so that some of the sections contain car images similar to what the classifier saw in the training set.

I decided to use the hog subsampling technique where the hog features were computed on the entire image for each video frame, then I took subsamples of different window sizes. This greatly sped up the speed of my pipeline. 

The box sizes I used in my final pipeline were as follows:
(list of parameters)

Here are some visualizations of each size:
![Alt text](.png?raw=True)
![Alt text](.png?raw=True)
![Alt text](.png?raw=True)
(test image with each box size)


--------------------------------------------------------------------------------
## Step 5. Run each segment through trained classifier
The final step in the pipeline is to run each of the subsections created in step 4 through the classifier. This will effectively classify each section as a car or as background. A box is then drawn around each section that is classified as a car.

The 2 most important factors in the success of the pipeline are the performance of the classifier and the window sizes that are chosen. The performance of the classifier is largely determined by the features chosen to train on so tweaking the feature vectors is a big factor in improving classifier performance. But even if the classifer is perfect, if the window sizes are way too small or way too big and are not at all similar to the training data then the classifier will have a hard time making proper distinctions. So experimenting with different window sizes is another big factor in the performance of the pipeline.

----------------------------------------------------------------------------------
Since we can't expect to have a perfect classifier nor perfect window sizes, we have to take some measures to correct for some errors in classification. This is where I used heatmaps. The heatmapping technique helps to correct for the errors caused by the imperfections described above. False positives from an imperfect classifier and overlapping window detections from imperfect window sizes. 

Heatmapping works as follows:
We create an array of 0's the same size as the image. Once one of our subsections is classified as a vehicle, we add one to this entire section of the heat map. Since our window sizes are overlapping (see step 4 visualizations) a good classifier should produce detections that look like this:
![Alt text](.png?raw=True) (test image with detection boxes drawn)

Adding one for each section would result in a heatmap that looks like this, where the redness indicates the 'heat' of that section:
![Alt text](.png?raw=True) (heatmap of above image)

Using the scipy.ndimage.measurements.label() function I am able to identify the primary sections contained in the heatmap and separate the blobs of heat. This ideally corresponds to each separate vehicle. I then pass all of the labels through the find_heat_boxes() function to extract the box corners for each blob (detected vehicle). These extracted box corners are the final boxes that I will draw onto each image which hopefully identifies all the vehicles in the image. The ideal box would be as tight around the vehicle as possible in order to represent the location as well as the length and height of the vehicle as accurately as possible.

The above process takes care of the imperfect window size problem by combining all of the positive detections around a vehicle. The next thing to take care of is false positives. 

Here is an example of some false positives that can be generated by the classifier:
![Alt text](.png?raw=True) (false positive box drawn on test image)
![Alt text](.png?raw=True) (heatmap of above image)

As we can see from the heatmap, the false positive area is not as hot as the correct detections. We can get rid of this with a simple threshold. If any section of the image is not considered hot enough, we will disregard it.

The heatmapping techinique assumes that the pipeline is good enough to where there are multiple boxes on the vehicles and limited false positives. If either of these is out of hand, the heatmapping technique may fail and something else in the pipeline must be edited.




**Provide more references to the code**
**add images and explain each step even more simply if possible**
**re-organize heatmap section**
**explain heatmap adding summing**
# Pneumonia Image Classification using Convolutional Neural Network

## Data

## Data Cleaning
We removed 217 images that did not have a proper encoding. This left us with 5,053  files in our training set.  
Out of which, we set 20% from each class as a validation set.  

Our final training set included 1,042 normal chest X-ray images and 3,001 chest X-ray of pneumonia patients. We can see that there is a slight class imbalance in our data.  

## Exploratory Data Analysis
---
![example images](/PNG/example_images.png)  
From looking at a few randomly chosen samples, we can generally observe a bit of cloud around the lung/heart area from X-rays of pneumonia patients.

### Average X-Rays
![average_normal](/PNG/average_normal.png) ![average_pneumonia](/PNG/average_pneumonia.png)  
We calculated the average image of each class by using average pixel values after rescaling all items to be 64x64 pixels. We can see that visilbity of a heart sets two classes apart. 

### Difference Between Classes
![contrast](/PNG/contrast.png)  
Then we computed the difference between the average images of classes. We can again see that the edges that surround and define the heart area shows a big difference. (Red indicates lighter in normal and blue indicates lighter in pneumonia)

### Variability
![std_normal](/PNG/std_normal.png) ![std_pneumonia](/PNG/std_pneumonia.png)  
Then we calculated the standard deviation for each pixel (after rescaling to 64x64) to show which area was the most variable in either class. Here lighter area indicates the higher variability. Again we can see the clear contrast of the lung area and the edge around the heart in normal patients.

### Eigenimages
![eigen_set_normal](/PNG/eigen_set_normal.png)  
Lastly we applied the Principal Component Analysis (PCA) to our images to find dimensions that best explain either class. Here we are visualizing components that explay 70% of variability. (28 PCs for normal class) We can see that many detects the approximate definiton of ribcage and contrast denoting the location of the heart. 

![eigen_set_pneumonia](/PNG/eigen_set_pneumonia.png)  
Here we are seeing the 14 principal components that explain 70% of variability in pneumonia class. We can clearly see that the edge definition is lacking compared to the normal class.



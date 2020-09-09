# Detecting Pneumonia from X-Rays using Deep Learning
Eunjoo Byeon and Aren Carpenter

## Introduction
Pneumonia is an acute respitory bacterial or viral infection that inflames air sacs in the lungs. This condition is especially dangerous for the young and old as well as patients with underlying conditions. Left untreated, pneumonia can cause fevers, chills, and difficulty of breathing that can eventually lead to death. In fact, pneumonia is the eighth leading cause of death in the United States, and it is the leading cause of death worldwide for children under five years old, accounting for 1.4 million deaths a year (WHO report).

Compared to other ailments of equal morbidity, pneumonia is cheap and simple to treat, often just required antibiotics. The difficulty stems from a lack of medical infastructure, both equipment and personnel, especially in the hardest hit areas like South Asia and sub-Saharan Africa. Chest X-rays are a popular and cheap test that can effectively identify pneumonia, but it still requires a trained physician to correctly diagnosis. Hence, we describe a convolutional neural network that can identify the presence of pneumonia from X-rays alone and with great accuracy and recall.

## Data
Our dataset consisted of about 5000 labeled chest x-rays provided by Kermany, Zhang, Goldbaum, et al. as part of their article published in Cell. There was class imbalance typical of medical imaging of 1:3 normal to pneumonia.

### Data Cleaning
We removed 217 images that did not have a proper encoding. This left us with 5,053  files in our training set.  
Out of which, we set 20% from each class as a validation set.  

Our final training set included 1,042 normal chest X-ray images and 3,001 chest X-ray of pneumonia patients. 

## Exploratory Data Analysis
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


## Model Evaluation
### Evaluation Metrics
We used accuracy as the primary evaluation metric. Additionally we used a recall score as a supplementary metric because we would rather have a false positive that future testing would catch than a missed case.

### Loss Function
As this is a binary classification problem (Presence of Pneumonia or not) we used binary crossentropy as our loss function.
 
### Optimization 
We tested RMS-prop, Adam and Adam with AMSGrad algorithms. Using Adam-based optimizer was shown to be more optimal than RMS-Prop.

### Class Imbalance
When we are not expanding the dataset using data augmentation, we tested balancing out the class weight during model fitting. This slightly improved our validation accuracy. 

We developed a convolutional neural network with __ Conv2D layers with ReLu activation and batch normalization techniques before feeding into __ Dense layers.


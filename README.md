# Detecting Pneumonia from X-Rays using Deep Learning
Eunjoo Byeon and Aren Carpenter

## Introduction
Pneumonia is an acute respitory bacterial or viral infection that inflames air sacs in the lungs. This condition is especially dangerous for the young and old as well as patients with underlying conditions. Left untreated, pneumonia can cause fevers, chills, and difficulty of breathing that can eventually lead to death. In fact, pneumonia is the eighth leading cause of death in the United States, and it is the leading cause of death worldwide for children under five years old, accounting for 1.4 million deaths a year (WHO report).

Compared to other ailments of equal morbidity, pneumonia is cheap and simple to treat, often just required antibiotics. The difficulty stems from a lack of medical infastructure, both equipment and personnel, especially in the hardest hit areas like South Asia and sub-Saharan Africa. Chest X-rays are a popular and cheap test that can effectively identify pneumonia, but it still requires a trained physician to correctly diagnosis. Hence, we describe a convolutional neural network that can identify the presence of pneumonia from X-rays alone and with great accuracy and recall.

## Data
Our dataset consisted of about 5000 labeled chest x-rays provided by Kermany and his colleague as part of their article published in Cell. There was class imbalance typical of medical imaging of 1:3 normal to pneumonia.
*Kermany, Daniel; Zhang, Kang; Goldbaum, Michael (2018), “Large Dataset of Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images”, Mendeley Data, V3*

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
Since we would rather have false positive than to miss a pneumonia case in future testing, we would prioritize the recall score. But our dataset is imbalanced with the pneumonia (positive) case being majority, so the recall alone will not be enough to evaluate the model. (The recall will likely be good if our model prioritize answering positive) So we looked at accuracy and recall together. 

### Loss Function
As this is a binary classification problem (presence of pneumonia or not) we used binary crossentropy as our loss function.
 
### Optimization 
We tested RMS-prop, Adam and Adam with AMSGrad algorithms. Using Adam-based optimizer was shown to be more optimal than RMS-Prop.

### Class Imbalance
When we are not expanding the dataset using data augmentation, we tested balancing out the class weight during model fitting. This slightly improved our validation accuracy. Otherwise we assumed that data augmentation added enough data to account for the imbalance.

### Normalization
After fitting all X-Ray images into a square (either 150 or 200px), we rescaled each pixel to be between 0 to 1.

### Baseline Model
We developed an overfitting baseline convolutional neural network with 4 Conv2D layers with Relu activation and 3x3 filter followed by MaxPooling with 2x2 filter and stride of 2 before feeding into Dense layers. The baseline had a vert high recall but the loss was pretty high as well. 

| Model | Loss | Accuracy | Recall |
| ---- | ---- | ---- | ---- |
| Baseline | 3.75 | 0.83 | 0.99 |

### Iterative Process
We took the iterative process to develop our model. Our process included adjusting the complexity of the model, fine tuning data augmentation criteria, using batch normalization and pre-trained network. 

### Final Model
Below table and image shows the architecture of our best performing model.  

![final model](/PNG/final_model_architecture.png)  

| Layer (type) | Output Shape | Param # | 
| ---- | ---- | ---- |
| conv2d_94 (Conv2D) | (None, 150, 150, 32) | 320 |   
| max_pooling2d_94 (MaxPooling | (None, 75, 75, 32) | 0 |
| conv2d_95 (Conv2D) | (None, 75, 75, 64) | 18496 |
| max_pooling2d_95 (MaxPooling | (None, 37, 37, 64) | 0 |
| conv2d_96 (Conv2D) | (None, 37, 37, 256) | 147712 |
| max_pooling2d_96 (MaxPooling | (None, 18, 18, 256) | 0 |
| flatten_25 (Flatten) | (None, 82944) | 0 |
| dense_51 (Dense) | (None, 1024) | 84935680 |
| dense_52 (Dense) | (None, 1) | 1025 |

#### Performance
Our final model showed the accuracy of 95% in classifying between pneumonia and normal case. It captured 97% of pneumonia cases.

| Model | Loss | Accuracy | Recall |
| ---- | ---- | ---- | ---- |
| Baseline | 3.75 | 0.83 | 0.99 |
| Final | 0.25 | 0.95 | 0.97 |

![final model_confusion matrix](/PNG/confusion_matrix.png)  


### Evaluation
We then looked at where our model failed.  

![FN image](/PNG/FN_image.png)  ![FP_image](/PNG/FP_image.png)  

The image on the left is X-ray of pneumonia patient, which our model classified as normal. The image on the right is from healthy patient, which our model classified to have pneumonia. We can suspect that the model fails to detect pneumonia when pneumonia is not significantly obstructing the view of ribcage and other organs. Also it may perform poorly when the X-ray image of healthy patient has a low contrast. We might benefit from increasing overall input size.

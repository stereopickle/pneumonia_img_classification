# Detecting Pneumonia from X-Rays with Deep Learning

#### Eunjoo Byeon and Aren Carpenter
#### DS Cohort 062220

### This Repo Contains:

File names here...

## Introduction

Pneumonia is an acute respitory bacterial or viral infection that inflames air sacs in the lungs. This condition is especially dangerous for the young and old as well as patients with underlying conditions. Left untreated, pneumonia can cause fevers, chills, and difficulty of breathing that can eventually lead to death. In fact, pneumonia is the eighth leading cause of death in the United States, and it is the leading cause of death worldwide for children under five years old, accounting for 1.4 million deaths a year (WHO report). 

Compared to other ailments of equal morbidity, pneumonia is cheap and simple to treat, often just required antibiotics. The difficulty stems from a lack of medical infastructure, both equipment and personnel, especially in the hardest hit areas like South Asia and sub-Saharan Africa. Chest X-rays are a popular and cheap test that can effectively identify pneumonia, but it still requires a trained physician to correctly diagnosis. Hence, we describe a convolutional neural network that can identify the presence of pneumonia from X-rays alone and with great accuracy and recall. 

## Data and Modeling

Our dataset consisted of about 5000 labeled chest x-rays provided by Kermany, Zhang, Goldbaum, et al. as part of their article published in Cell. There was class imbalance typical of medical imaging of 1:3 normal to pneumonia. 

We developed a convolutional neural network with __ Conv2D layers with ReLu activation and batch normalization techniques before feeding into __ Dense layers. As this is a binary classification problem (Presence of Pneumonia or not) we used binary crossentropy as our loss function. After some experimentation, we decided on __ as our optimizer instead of __ because __ "cite literature". We used accuracy and recall as our evaluation metrics because we would rather have a false positive that future testing would catch than a missed case. 

## Results and Insights



## Future Directions

Future directions include...
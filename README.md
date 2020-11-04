![tittle](images/face_app_cover.jpg)

# Face_Shape_Classification
This is an image classification project to identify 5 female face shapes using Convolutional Neural Networks.  I completed this as my Capstone Project for Data Science Immersive course with General Assembly (October 2020).

# Problem Statement

Based on [The Deloitte Consumer Review](https://www2.deloitte.com/content/dam/Deloitte/ch/Documents/consumer-business/ch-en-consumer-business-made-to-order-consumer-review.pdf), consumers are demanding more personalised experience, however trial remains low.  In beauty and fashion industry, over 40% of adults aged 16-39 are interested in personalised offer, while trial is only 10% and 14% respectively.  Among those who are interested, around 80% are willing to pay premium price (10% or more). 

By being able to classify face shapes will enable brands to offer more personalised solutions to better satisfy their consumers, while increasing margin from premium positioning.  Example of use cases are:
- Customised face mask 
- Customised make-up tips based on face shape [vdo](https://www.youtube.com/watch?v=ZeJP_2IM-2Q)
- Recommendations for glasses or sunglasses
- Recommendations for fashion accessories (i.e. earrings, necklaces, hat)

For this project, I will be using Deep Learning approach with Convolutional Neural Networks to classify 5 different female face shapes (Heart, Oblong, Oval, Round, Square).  The model that was highest accuracy score will be chosen.

# Data

The [Face Shape Dataset](https://www.kaggle.com/niten19/face-shape-dataset) is a dataset from Kaggle by Niten Lama. 

This dataset comprises a total of 5000 images of the female celebrities from all around the globe which are categorized according to their face-shape namely: 
- Heart
- Oblong
- Oval 
- Round
- Square

Each category consists of 1000 images (800 for training : 200 for testing)

## Conclusion & Key Learning:

The CNN models are suitable for image classification as it trains well with fewer parameter when compared to Fully Connected model.  The CNN model from scratch achieves accuracy of ~80%, while incorporating transfer learning using VGG-Face increased the accuracy to over 90%. 

Key learning:
- Image pre-process (specifically the bounding box) helps the model train better especially with limited dataset (4000 training images).
- Transfer Learning with the weights trained on larger datasets (2.6 million images) can help improve the accuracy of training as well as the speed.  
- Oval shape is harder to predict with misclassification in different classes.  
- The model performs better on images where the bounding box can be easily detected, however does not predict well on images that the full face cannot be detected (i.e. circle frame, or tilted angle).  



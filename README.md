1[](/images/face_app_cover.jpg)

# Face_Shape_Classification
This project is my capstone project as part of my Data Science Immersive course at General Assembly.  It aims to classify facial images into 5 different face shapes with Convolutional Neural Networks.  

# Problem Statement

Based on [The Deloitte Consumer Review](https://www2.deloitte.com/content/dam/Deloitte/ch/Documents/consumer-business/ch-en-consumer-business-made-to-order-consumer-review.pdf), consumers are demanding more personalised experience, however there is still an unmet need in some categories, especially beauty and fashion industry.  Over 40% of adults aged 16-39 are interested in personalised beauty products and fashion accessories/jewelry, while trial remains low (only 10% and 14% respectively).  Among those who are interested in personalised products for these 2 categories, around 80% are willing to pay premium price (10% or more). 

By being able to classify face shapes, brands will be able to offer more personalised solutions to better satisfy their consumers, while increasing margin from premium positioning.  Example of use cases are:
- Customised face mask 
- Customised make-up tips based on face shape [vdo](https://www.youtube.com/watch?v=ZeJP_2IM-2Q)
- Recommendations for glasses or sunglasses
- Recommendations for accessories (i.e. earrings)

For this project, we will be using Deep Learning approach with Convolutional Neural Networks to classify 5 different female face shapes (Heart, Oblong, Oval, Round, Square).  The model that was highest accuracy score will be chosen.

## Conclusion & Key Learning:

The CNN models are suitable for image classification as it trains well with fewer parameter when compared to Fully Connected model.  The CNN model from scratch achieves accuracy of ~80%, while incorporating transfer learning using VGG-Face increased the accuracy to over 90%. 

Key learning:
- Image pre-process (specifically the bounding box) helps the model train better especially with limited dataset (4000 training images).
- Transfer Learning with the weights trained on larger datasets (2.6 million images) can help improve the accuracy of training as well as the speed.  
- Oval shape is harder to predict with misclassification in different classes.  
- The model performs better on images where the bounding box can be easily detected, however does not predict well on images that the full face cannot be detected (i.e. circle frame, or tilted angle).  



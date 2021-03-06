# Implicit-Rating-Predictions-on-booking.com-data
Experiment with Machine Learning Model that can predict whether a user will prefer a trip destination recommendation given the historical user-item interaction data.

The aim of this project was to experiment with different Machine Learning models for generating recommendations. We experimented with ensembles of Matrix Factoriztion, Decision Trees, Random Forests, Neural Networks and XGBoost algorithms to predict the implicit ratings given by user to a recommended trip destination.

## About the Data

Our data contains:
1. User id: A unique id specific to each user. (Total users = 169698)
2. Hotel id: A unique id specific to each hotel item. (Total unique items = 39001)
3. Device type id: An encoded categorical variable representative the type of device used by users. It has 4 unique values.
4. City id: An encoded categorical variable representative of different locations of the hotel-id. It has 195 unique values.

## Data Preprocessing Workflow

![Data Preprocessing](https://github.com/ashwanirajan/Implicit-Rating-Predictions-on-booking.com-data/blob/main/preprocessing.jpg)

## Negative Sampling

1. **Random Negative Sampling**: Randomly choose N user id-item id pairs from
the training data and assign them zero label. User behaviour is not
encapsulated in this case.
2. **User-based Random Sampling**: For each user, generate either equal
negative samples as positive samples or generate negative samples in the
ratio 1:2 or 1:3.
3. **Inverse Popularity Negative Sampling**: Normalize the data using min-max
scaler and subtract each score from 1. Use the result as weights for each
score and generate negative data using these weights

## Modeling
Matrix Factorization             |  Model Architecture
:-------------------------|:-------------------------:
Initialize user, item, feature and item embeddings of appropriate shapes.<br>Concatenate user embedding with context feature embeddings and item embeddings with item feature embeddings for all training entries.<br>Matrix multiplication of the concatenated user and item matrices.<br>Sigmoid returns the probability of getting the output label=1, i.e. in this case, probability whether user likes the item. <br>Binary Cross Entropy loss is used for this problem|  ![](https://github.com/ashwanirajan/Implicit-Rating-Predictions-on-booking.com-data/blob/main/MF_model.jpg)

**RESULTS**
![Results](https://github.com/ashwanirajan/Implicit-Rating-Predictions-on-booking.com-data/blob/main/results1.jpg)

Neural Network             |  Model Architecture
:-------------------------|:-------------------------:
Initialize user, item, feature and item embeddings of appropriate shapes.<br>Concatenate all embeddings for any training entry.<br>Linear layer 1 takes input size equal to length of concatenated vector and returns an intermediate layer.<br> Dropout with probability of 0.2-0.3 is added to prevent overfitting.<br> Sigmoid returns the class probabilities, in this case, the probability of user liking item.|  ![](https://github.com/ashwanirajan/Implicit-Rating-Predictions-on-booking.com-data/blob/main/NN.jpg)

**RESULTS**
![Results](https://github.com/ashwanirajan/Implicit-Rating-Predictions-on-booking.com-data/blob/main/results2.jpg)

Tree-Based Models             |  Model Architecture
:-------------------------|:-------------------------:
Initialize user and item embeddings of appropriate shapes(embedding size = 50).<br> Train the embeddings in a Matrix Factorization model. <br> Concatenate the trained user and item embeddings for all training entries.<br> The concatenated vectors are passed to tree based models, along with the encoded context and item feature ids.<br> model.predict_proba() returns the class probabilities for user liking and disliking the item.|  ![](https://github.com/ashwanirajan/Implicit-Rating-Predictions-on-booking.com-data/blob/main/tree_models.jpg)

**RESULTS**
![Results](https://github.com/ashwanirajan/Implicit-Rating-Predictions-on-booking.com-data/blob/main/results3.jpg)
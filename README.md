# Oil and Gas Prediction Machine Learning

## Dataset
This folder contains the dataset that will be used for this study.

The original folder contains the original dataset obtained from online sources. The Volve production dataset was obtained from Kaggle and 
can be accessed through this [link](https://experience.arcgis.com/experience/50b61d215bff4072bf0649efe6e8d). On the other hand,
the Kyle dataset was obtained from the online data centre of the Oil and Gas Authority and can be accessed through this [link](https://experience.arcgis.com/experience/50b61d215bff4072bf0649efe6e8d). The Kyle dataset consists of data from four different wellbores.

The cleaned folder contains the dataset that have been cleaned and processed in three different ways. The first way is by using the forward filling method
to fill in the missing values, the second method is by using the median imputation method to fill in the missing values. The third method is by using a
machine learning model to fill in the missing values in the dataset. 

The testing folder contains the test dataset which will be used to test the perfomance of the model. As the the Kyle dataset contained data from four
different wellbores, the data from one of these wellbores was used as the test dataset.

## Code
This folder contains the code for the Exploratory Data Analysis of the dataset. It also contains the code for the Gradient Boosting and Random Forest models. Additionally, the code for the hyperparameter optimization method using HyperOpt is also provided. 

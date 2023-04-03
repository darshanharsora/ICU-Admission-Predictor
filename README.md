# ICU-Admission-Predictor
## Introduction
Coronavirus disease (COVID-19) is an infectious disease caused by the SARS-CoV-2 virus. Most virus-infected individuals will have a mild to severe respiratory disease and will recover without the need for special care. However, some people will get serious illnesses and need to see a doctor. Serious sickness is more likely to strike older persons and those with underlying medical illnesses including cancer, diabetes, cardiovascular disease, or chronic respiratory diseases. COVID-19 can cause anyone to get very ill or pass away at any age.

The COVID-19 pandemic affected the whole world, overwhelming hospital institutions that were not equipped to handle the prolonged and strong demand for ICU beds, staff, personal safety gear, and medical resources. In 2020, the world entered the period of communal transmission.

A significant increase of cases is difficult since it puts a lot of strain on the local hotspot's health care system. Only a tiny number of COVID-19 patients require intensive care. Therefore, the need for ICU beds increases as the number of patients increases. But sometimes the financially stable section of the society who are suffering from Covid-19 book ICU before any need arises, thus blocking the access to ICU to people who are really in need. So, if we can predict whether a patient will need an ICU bed or not may help saving many lives as we can make the arrangements prior if the situation worsens. So we developed three different machine learning models which can predict the requirement of ICU for Covid-19 patient.

## Dataset
The Covid-19 ICU admission dataset includes aggregated data of 5,66,002 Covid-19 patients who were treated at various hospitals throughout the world. In accordance with the finest international standards and guidelines, all data were kept anonymous. The dataset was taken from GitHub (https://github.com/paulanderson7772/machinelearning) where the author made it available to public for development. The dataset has 22 features like date of admission, age, information about various diseases other than covid (Co-Morbid), etc. The target feature is “icu” which states whether a patient will need admission in ICU or not.

By exploration of the dataset, we have found that there are no null values in the dataset, but some features have many instances with garbage value which will be dropped. Various visualization of the dataset provided us with important insights about scaling and normalizing data. Many features were found to be irrelevant in our development, so those features were dropped after examining their importance.

## Methodology
In our approach to solve this problem, we applied various supervised machine learning algorithms like Random Forest Classifier, K Nearest Neighbour Classifier and Support Vector Machine (SVM) as we have discrete labelled data. The Data is having redundancies so, the data needs to be cleaned and pre-processed before feeding to the model for training. Many unwanted features like ‘id’, ‘contact_other_covid’, etc. were dropped as they have no relevance with the admission of person in ICU.
The dataset is imbalanced, so we will implement SMOTE to overcome this problem. We will compare the various ML models with the help of evaluation metrics viz. 1) Accuracy 2) Precision 3) Recall 4) F1 Score which in turn will help us in choosing the best model for prediction. Also, we will be tuning the hyperparameters to improve the performance of the model using GridSearchCV. By constructing a confusion matrix, we will be able to check for True Positives, True Negative, False Positives and False Negatives. The technique is based on doing several trials utilising the indicated algorithms, Random forest, KNN, and SVM, on a dataset. For the greatest accuracy, precision, recall, and F1 Score, unique trials are run on each method and on combinations of them.

## Results
Three algorithms were separately trained and the best model in terms of evaluation metrics was Random Forest with accuracy of 83.4%. Random Forest has a good True Positivity Rate, which results in a goof F1 score. KNN gave accuracy of 76.5%, with precision score of 71.3%, recall 89.2% which is acceptable. SVM had least accuracy of 66.9%, precision 64.4%, recall 76.8% and lowest F1 score of 70.1%.

## Author
-Darshan Harsora

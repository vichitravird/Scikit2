# Scikit2

1 Problem 1 : Deploy a Machine Learning Model using Streamlit Library ( https://www.geeksforgeeks.org/deploy-a-machine-learning-model-using-streamlit-library/)
import pandas as pd 
import numpy as np 

df = pd.read_csv('BankNote_Authentication.csv') 
df.head() 
# Dropping the Id column 
df.drop('Id', axis = 1, inplace = True) 

# Renaming the target column into numbers to aid training of the model 
df['Species']= df['Species'].map({'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2}) 

# splitting the data into the columns which need to be trained(X) and the target column(y) 
X = df.iloc[:, :-1] 
y = df.iloc[:, -1] 

# splitting data into training and testing data with 30 % of data as testing data respectively 
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0) 

# importing the random forest classifier model and training it on the dataset 
from sklearn.ensemble import RandomForestClassifier 
classifier = RandomForestClassifier() 
classifier.fit(X_train, y_train) 

# predicting on the test dataset 
y_pred = classifier.predict(X_test) 

# finding out the accuracy 
from sklearn.metrics import accuracy_score 
score = accuracy_score(y_test, y_pred) 

# pickling the model 
import pickle 
pickle_out = open("classifier.pkl", "wb") 
pickle.dump(classifier, pickle_out) 
pickle_out.close()

import pandas as pd 
import numpy as np 
import pickle 
import streamlit as st 
from PIL import Image 

# loading in the model to predict on the data 
pickle_in = open('classifier.pkl', 'rb') 
classifier = pickle.load(pickle_in) 

def welcome(): 
	return 'welcome all'

# defining the function which will make the prediction using 
# the data which the user inputs 
def prediction(sepal_length, sepal_width, petal_length, petal_width): 

	prediction = classifier.predict( 
		[[sepal_length, sepal_width, petal_length, petal_width]]) 
	print(prediction) 
	return prediction 
	

# this is the main function in which we define our webpage 
def main(): 
	# giving the webpage a title 
	st.title("Iris Flower Prediction") 
	
	# here we define some of the front end elements of the web page like 
	# the font and background color, the padding and the text to be displayed 
	html_temp = """ 
	<div style ="background-color:yellow;padding:13px"> 
	<h1 style ="color:black;text-align:center;">Streamlit Iris Flower Classifier ML App </h1> 
	</div> 
	"""
	
	# this line allows us to display the front end aspects we have 
	# defined in the above code 
	st.markdown(html_temp, unsafe_allow_html = True) 
	
	# the following lines create text boxes in which the user can enter 
	# the data required to make the prediction 
	sepal_length = st.text_input("Sepal Length", "Type Here") 
	sepal_width = st.text_input("Sepal Width", "Type Here") 
	petal_length = st.text_input("Petal Length", "Type Here") 
	petal_width = st.text_input("Petal Width", "Type Here") 
	result ="" 
	
	# the below line ensures that when the button called 'Predict' is clicked, 
	# the prediction function defined above is called to make the prediction 
	# and store it in the variable result 
	if st.button("Predict"): 
		result = prediction(sepal_length, sepal_width, petal_length, petal_width) 
	st.success('The output is {}'.format(result)) 
	
if __name__=='__main__': 
	main() 


2 Problem 2 : Machine Learning Model with Teachable Machine	(	https://www.geeksforgeeks.org/machine-learning-model-with-teachable-machine/)
Gather data: Use a web camera to gather images or videos of the objects you want to classify.
Train the model: Use the gathered images or videos to train a machine learning model. Teachable Machine provides a simple interface for labeling the images and training the model.
Test the model: Use the trained model to classify new images or videos, and see how accurate the model is.
Fine-tune the model: If the model is not accurate enough, you can go back and fine-tune it by gathering more data or adjusting the training parameters.
Teachable Machine is a great tool for individuals, educators and students to learn the basics of machine learning and computer vision. It can be used to train simple models for a wide range of applications such as image classification, object detection and gesture recognition.
Machine Learning and Artificial Intelligence have raised the level of applications. Many organizations are working on Artificial Intelligence to create an impact on society. Machine learning is the backbone of Artificial Intelligence. But everyone doesn’t know how machine learning works and how to create models that can be used in artificial intelligence. Don’t worry; it is possible now. You might be wondering how? It is super easy for non-coders or coders not from machine learning backgrounds to create a machine learning model and integrate it within the application. In this article, we are going to build a machine learning model without coding a single line.


3 Problem 3 : Implementing PCA in Python with scikit-learn	(	https://www.geeksforgeeks.org/implementing-pca-in-python-with-scikit-learn/)
# import all libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#import the breast _cancer dataset
from sklearn.datasets import load_breast_cancer
data=load_breast_cancer()
data.keys()

# Check the output classes
print(data['target_names'])

# Check the input attributes
print(data['feature_names'])

# Check the values of eigen vectors
# prodeced by principal components
principal.components_

# Check the values of eigen vectors
# prodeced by principal components
principal.components_

plt.figure(figsize=(10,10))
plt.scatter(x[:,0],x[:,1],c=data['target'],cmap='plasma')
plt.xlabel('pc1')
plt.ylabel('pc2')

# import relevant libraries for 3d graph
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(10,10))

# choose projection 3d for creating a 3d graph
axis = fig.add_subplot(111, projection='3d')

# x[:,0]is pc1,x[:,1] is pc2 while x[:,2] is pc3
axis.scatter(x[:,0],x[:,1],x[:,2], c=data['target'],cmap='plasma')
axis.set_xlabel("PC1", fontsize=10)
axis.set_ylabel("PC2", fontsize=10)
axis.set_zlabel("PC3", fontsize=10)

# check how much variance is explained by each principal component
print(principal.explained_variance_ratio_)




4 Problem 4 : Pipelines – Python and scikit-learn	(	https://www.geeksforgeeks.org/pipelines-python-and-scikit-learn/)
Gathering data: 
The process of gathering data depends on the project it can be real-time data or the data collected from various sources such as a file, database, survey and other sources.
Data pre-processing: 
Usually, within the collected data, there is a lot of missing data, extremely large values, unorganized text data or noisy data and thus cannot be used directly within the model, therefore, the data require some pre-processing before entering the model.
Training and testing the model: Once the data is ready for algorithm application, It is then ready to put into the machine learning model. Before that, it is important to have an idea of what model is to be used which may give a nice performance output. The data set is divided into 3 basic sections i.e. The training set, validation set and test set. The main aim is to train data in the train set, to tune the parameters using ‘validation set’ and then test the performance test set.
Evaluation: 
Evaluation is a part of the model development process. It helps to find the best model that represents the data and how well the chosen model works in the future. This is done after training of model in different algorithms is done. The main motto is to conclude the evaluation and choose model accordingly again.

# deep-learning-challenge

## Step 1: Preprocess the Data
Using your knowledge of Pandas and scikit-learn’s StandardScaler(), you’ll need to preprocess the dataset. This step prepares you for Step 2, where you'll compile, train, and evaluate the neural network model.

Start by uploading the starter file to Google Colab, then using the information we provided in the Challenge files, follow the instructions to complete the preprocessing steps.

Read in the charity_data.csv to a Pandas DataFrame, and be sure to identify the following in your dataset:
What variable(s) are the target(s) for your model?
What variable(s) are the feature(s) for your model?
Drop the EIN and NAME columns.

Determine the number of unique values for each column.

For columns that have more than 10 unique values, determine the number of data points for each unique value.

Use the number of data points for each unique value to pick a cutoff point to bin "rare" categorical variables together in a new value, Other, and then check if the binning was successful.

Use pd.get_dummies() to encode categorical variables.

Split the preprocessed data into a features array, X, and a target array, y. Use these arrays and the train_test_split function to split the data into training and testing datasets.

Scale the training and testing features datasets by creating a StandardScaler instance, fitting it to the training data, then using the transform function.

## Step 2: Compile, Train, and Evaluate the Model
Using your knowledge of TensorFlow, you’ll design a neural network, or deep learning model, to create a binary classification model that can predict if an Alphabet Soup-funded organization will be successful based on the features in the dataset. You’ll need to think about how many inputs there are before determining the number of neurons and layers in your model. Once you’ve completed that step, you’ll compile, train, and evaluate your binary classification model to calculate the model’s loss and accuracy.

Continue using the file in Google Colab in which you performed the preprocessing steps from Step 1.

Create a neural network model by assigning the number of input features and nodes for each layer using TensorFlow and Keras.

Create the first hidden layer and choose an appropriate activation function.

If necessary, add a second hidden layer with an appropriate activation function.

Create an output layer with an appropriate activation function.

Check the structure of the model.

Compile and train the model.

Create a callback that saves the model's weights every five epochs.

Evaluate the model using the test data to determine the loss and accuracy.

Save and export your results to an HDF5 file. Name the file AlphabetSoupCharity.h5.

## Step 3: Optimize the Model
Using your knowledge of TensorFlow, optimize your model to achieve a target predictive accuracy higher than 75%.

Use any or all of the following methods to optimize your model:

Adjust the input data to ensure that no variables or outliers are causing confusion in the model, such as:
Dropping more or fewer columns.
Creating more bins for rare occurrences in columns.
Increasing or decreasing the number of values for each bin.
Add more neurons to a hidden layer.
Add more hidden layers.
Use different activation functions for the hidden layers.
Add or reduce the number of epochs to the training regimen.
Note: If you make at least three attempts at optimizing your model, you will not lose points if your model does not achieve target performance.

Create a new Google Colab file and name it AlphabetSoupCharity_Optimization.ipynb.

Import your dependencies and read in the charity_data.csv to a Pandas DataFrame.

Preprocess the dataset as you did in Step 1. Be sure to adjust for any modifications that came out of optimizing the model.

Design a neural network model, and be sure to adjust for modifications that will optimize the model to achieve higher than 75% accuracy.

Save and export your results to an HDF5 file. Name the file AlphabetSoupCharity_Optimization.h5.

## Result 
** The target variable for the model is the "IS_SUCCESSFUL" column.

** The features used in the model are 'APPLICATION_TYPE', 'INCOME_AMT', 'CLASSIFICATION', 'AFFILIATION', 'ORGANIZATION', 'USE_CASE', and 'SPECIAL_CONSIDERATIONS'.

** The 'SPECIAL_CONSIDERATIONS' variable was removed in the optimized model as it was deemed insignificant to predicting the target variable.
Compiling, Training, and Evaluating the Model:

For the optimized model, 74 neurons were utilized per hidden layer, chosen because 74 is 2 times the number of input dimensions (37). Three layers were employed, all featuring the hyperbolic tangent (tanh) activation function. Tanh was selected for its ability to incorporate negative values more effectively. Despite these efforts, the target model performance was not achieved. Several strategies were attempted, including:
Removing columns individually: 'USE_CASE', 'ORGANIZATION', 'AFFILIATION', 'STATUS', 'INCOME_AMT', and 'ASK_AMT'.
Dropping 'ORGANIZATION' and 'AFFILIATION' simultaneously.
Dropping 'INCOME_AMT' and 'ASK_AMT' simultaneously.
However, no significant improvement was observed when dropping any of these columns. Ultimately, only the 'SPECIAL_CONSIDERATIONS' column was dropped, as it appeared to be the least relevant for predicting the success of a venture.

Experimentation was also conducted with the number of neurons, testing an increase to 99 neurons per layer, but this did not yield substantial improvements.

Various combinations of activation functions were explored, such as (relu, tanh, tanh), (tanh, tanh, relu), and (relu, relu, tanh). The final configuration settled on using tanh activation for all three layers. Despite these attempts, the desired model performance remained elusive.

## Summary

The ultimately optimized neural network model, fine-tuned through the Keras Tuner method, exhibited remarkable performance with an 80% accuracy and a minimal loss of 0.45. Utilizing a sigmoid activation function, the model boasted a sophisticated architecture, including an input layer of 76 nodes and five hidden layers with 16, 21, 26, 11, and 21 neurons, respectively, culminating in 50 training epochs. Its superiority over the non-optimized model highlighted the efficacy of automated tuning. The decision to retain the 'Name' column emerged as a pivotal factor, playing a crucial role in surpassing performance targets. This outcome underscores the significance of thoughtful dataset structuring prior to preprocessing, showcasing the profound impact of data organization on final model effectiveness.

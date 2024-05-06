# Deep Learning: Predicting Success of Funded Organizations

# Introduction
This project focuses on building a binary classifier using neural networks to predict the success of organizations funded by Alphabet Soup, a nonprofit foundation. The dataset contains metadata about organizations funded by Alphabet Soup, such as application type, affiliation, classification, use case for funding, funding amount requested, and whether the money was used effectively.

# Guide
- EIN and NAME- Identification columns
- APPLICATION_TYPE- Alphabet Soup application type
- AFFILIATION- Affiliated sector of industry
- CLASSIFICATION- Government organization classification
- USE_CASE- Use case for funding
- ORGANIZATION- Organization type
- STATUS- Active status
- INCOME_AMT- Income classification
- SPECIAL_CONSIDERATIONS- Special considerations for application
- ASK_AMT- Funding amount requested
- IS_SUCCESSFUL- Was the money used effectively

# Instructions
# Step 1: Preprocess the Data
- Read the charity_data.csv into a Pandas DataFrame.
- Identify the target(s) and features for the model and drop unnecessary columns (EIN and NAME).
- Determine the number of unique values for each column and handle rare occurrences by combining them into a new category.
- Encode categorical variables using pd.get_dummies().
- Split the preprocessed data into features array (X) and target array (y), and then split the data into training and testing datasets.
- Scale the training and testing features datasets using StandardScaler.

# Step 2: Compile, Train, and Evaluate the Model
- Create a neural network model using TensorFlow and Keras.
- Design the model architecture with appropriate number of input features, neurons, and layers.
- Compile and train the model, and evaluate its performance using test data to calculate loss and accuracy.
- Save the model's weights every five epochs and export the results to an HDF5 file named AlphabetSoupCharity.h5.

# Step 3: Optimize the Model
- Experiment with various methods to optimize the model's performance.
- Adjust input data, add neurons or hidden layers, use different activation functions, and modify the number of epochs.
- Create a new Google Colab file (AlphabetSoupCharity_Optimization.ipynb) and preprocess the dataset.
- Design and train an optimized neural network model to achieve higher than 75% accuracy.
- Save the optimized model's results to an HDF5 file named AlphabetSoupCharity_Optimization.h5.

# Step 4: Write a Report on the Neural Network Model
- Write a report summarizing the analysis, results, and recommendations for improving the model.
- Provide an overview of the analysis and address questions related to data preprocessing and model compilation, training, and evaluation.
- Summarize the overall results of the deep learning model and recommend alternative models for solving the classification problem.

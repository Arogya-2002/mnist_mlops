## Mlops on MNIST dataset
1. Data ingestion:
    -> Created .npy files of mnist dataset in train and test variables
2. Data transformation:
    -> Performed train test split 
    -> Normalized the data
    -> Performed encoding
    -> Returns X_train,y_train,x_test,y_test
3. model trainer
    -> It takes in X_train,y_train,x_test,y_test and returns x_test,y_test for evaluation
4. Model evaluation
    -> Takes in x_test,y_test and evaluates model performance
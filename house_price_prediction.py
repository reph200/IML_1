import os

import numpy as np
import pandas as pd

# TODO: ChatGPT assistance
import matplotlib

from linear_regression import LinearRegression

matplotlib.use('Agg')  # Use a suitable non-interactive backend like Agg

from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
from typing import NoReturn


def preprocess_train(X: pd.DataFrame, y: pd.Series):
    """
    preprocess training data.
    Parameters
    ----------
    X: pd.DataFrame
        the loaded data
    y: pd.Series

    Returns
    -------
    A clean, preprocessed version of the data
    """

    # Drop specified columns from X
    X.drop(columns=['lat', 'long', 'id', 'date', 'yr_renovated'], inplace=True)

    # Combine X and y into a single DataFrame for easier preprocessing
    df = pd.concat([X, y], axis=1)

    # Remove rows with any NaN values from X and the corresponding entries in y
    df.dropna(inplace=True)

    # Remove rows with any negative values from X and the corresponding entries in y
    df = df[(df >= 0).all(axis=1)]

    # Replace 0 bedrooms or 0 bathrooms with the mean value of the respective columns in X
    mean_bedrooms = df['bedrooms'].mean()
    mean_bathrooms = df['bathrooms'].mean()

    # Explicitly cast the mean values to the same data type as the column
    mean_bedrooms = int(mean_bedrooms) if df['bedrooms'].dtype == 'int64' else mean_bedrooms
    mean_bathrooms = int(mean_bathrooms) if df['bathrooms'].dtype == 'int64' else mean_bathrooms

    df.loc[df['bedrooms'] == 0, 'bedrooms'] = mean_bedrooms
    df.loc[df['bathrooms'] == 0, 'bathrooms'] = mean_bathrooms

    # Remove rows where sqft_living15 is greater than sqft_lot15 from X and the corresponding entries in y
    df = df[df['sqft_living15'] < df['sqft_lot15']]

    # Remove rows where sqft_living is greater than sqft_lot from X and the corresponding entries in y
    df = df[df['sqft_living'] < df['sqft_lot']]

    # Add a new has_basement column to be 1 if the value of the sqft_basement column is not 0, otherwise stay 0 in X
    df['has_basement'] = df['sqft_basement'].apply(lambda x: 1 if x != 0 else 0)

    # Add a new has_been_viewed column to be 1 if the value the view column is not 0, otherwise stay 0 in X
    df['has_been_viewed'] = df['view'].apply(lambda x: 1 if x != 0 else 0)

    # Split the combined DataFrame back into X and y
    X = df.drop(columns=['price'])
    y = df['price']

    # Return the preprocessed feature data (X) and the response variable (y)
    return X, y


def preprocess_test(X: pd.DataFrame):
    """
    preprocess test data. You are not allowed to remove rows from X, but only edit its columns.
    Parameters
    ----------
    X: pd.DataFrame
        the loaded data

    Returns
    -------
    A preprocessed version of the test data that matches the coefficients format.
    """

    # All similar to the preprocess_train function except no rows are removed, as noted and required

    X.drop(columns=['lat', 'long', 'id', 'date', 'yr_renovated'], inplace=True)

    mean_bedrooms = X['bedrooms'].mean()
    mean_bathrooms = X['bathrooms'].mean()

    mean_bedrooms = int(mean_bedrooms) if X['bedrooms'].dtype == 'int64' else mean_bedrooms
    mean_bathrooms = int(mean_bathrooms) if X['bathrooms'].dtype == 'int64' else mean_bathrooms

    X.loc[X['bedrooms'] == 0, 'bedrooms'] = mean_bedrooms
    X.loc[X['bathrooms'] == 0, 'bathrooms'] = mean_bathrooms

    X['has_basement'] = X['sqft_basement'].apply(lambda x: 1 if x != 0 else 0)

    X['has_been_viewed'] = X['view'].apply(lambda x: 1 if x != 0 else 0)

    return X


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    # Ensure the output directory exists
    os.makedirs(output_path, exist_ok=True)

    for column in X.columns:
        # Compute the covariance between the feature and the response
        covariance = np.cov(X[column], y)[0, 1]

        # Compute the standard deviation of the feature and the response
        std_feature = np.std(X[column])
        std_response = np.std(y)

        # Compute the Pearson correlation coefficient, according to the formula given
        pearson_corr = covariance / (std_feature * std_response)

        # Create the scatter plot
        plt.figure(figsize=(8, 6))
        plt.scatter(X[column], y, alpha=0.5)
        plt.xlabel(column)
        plt.ylabel("Response")
        plt.title(f"{column} vs Response\nPearson Correlation: {pearson_corr:.2f}")

        # Save the plot to the specified output path
        file_name = f"{column}_vs_response.png"
        plt.savefig(os.path.join(output_path, file_name))
        plt.close()


if __name__ == '__main__':
    # Data source - https://www.kaggle.com/datasets/harlfoxem/housesalesprediction
    df = pd.read_csv("house_prices.csv")
    X, y = df.drop("price", axis=1), df.price

    random_state = np.random.randint(1000)

    # Question 2 - Split data to trian and test data frames according to the ratio given
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=random_state)

    # Question 3
    X_preprocessed_train, y_preprocessed_train = preprocess_train(X_train, y_train)

    # Question 4 - Feature evaluation of train dataset with respect to response
    feature_evaluation(X_preprocessed_train, y_preprocessed_train,
                       output_path="./features_vs_response_plots_for_training_data")

    # Question 5 - preprocess the test data
    X_preprocessed_test = preprocess_test(X_test)

    # Question 6 - Fit model over increasing percentages of the overall training data

    percentages = range(10, 101)

    # Number of repetitions for each percentage
    num_repetitions = 10

    # Lists to store average loss and variance
    average_losses = []
    variances = []

    for percentage in percentages:
        losses = []
        for _ in range(num_repetitions):
            # Set random state for consistent sampling
            random_state = np.random.randint(1000)

            # Sample p% of the overall training data using pandas.sample
            sample_size = int(len(X_preprocessed_train) * percentage / 100)
            X_sampled = X_preprocessed_train.sample(n=sample_size, replace=False, random_state=random_state)
            y_sampled = y_preprocessed_train.loc[X_sampled.index]

            # Fit linear model
            model = LinearRegression()
            model.fit(X_sampled, y_sampled)

            # Calculate loss
            loss = model.loss(X_preprocessed_test.values, y_test)
            losses.append(loss)

        # Store average and variance of loss over test set
        average_loss = np.mean(losses)
        variance = np.var(losses)

        average_losses.append(average_loss)
        variances.append(variance)

    percentage_ticks = list(percentages)

    # Plot average loss as function of training size with error bars
    plt.plot(percentage_ticks, average_losses, '-', label='Average Loss')
    plt.xlabel('Percentage of Training Data')
    plt.ylabel('Average Loss')
    plt.title('Average Loss (MSE) vs. Training Size')

    # Plot the confidence interval as a light grey continuous background
    # Similar to the scatter function from the lab, but shorter and more efficient
    plt.fill_between(percentage_ticks, np.array(average_losses) - 2 * np.sqrt(variances),
                     np.array(average_losses) + 2 * np.sqrt(variances), color='lightgrey', alpha=0.5,
                     label='Confidence Interval')

    # plt.show()
    plt.savefig('./average_loss_vs_training_size.png')
    plt.close()

    print("hello baby")

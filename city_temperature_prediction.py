import matplotlib
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split

from polynomial_fitting import PolynomialFitting

matplotlib.use('TkAgg')  # Use a suitable interactive backend like Agg

from matplotlib import pyplot as plt

COLDEST_TEMPERATURE = -20


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data. temprature
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    # Read the CSV file into a DataFrame, parsing the 'Date' column as datetime
    df = pd.read_csv(filename, parse_dates=["Date"])

    # Drop rows with missing values to clean the dataset
    df.dropna(inplace=True)

    # Filter out rows where the temperature is below a realistic minimum value
    df = df[df['Temp'] >= COLDEST_TEMPERATURE]

    # Add a new column 'DayOfYear' which is the day of the year extracted from the 'Date' column
    df["DayOfYear"] = df["Date"].dt.dayofyear

    return df


if __name__ == '__main__':
    # Question 2 - Load and preprocessing of city temperature dataset
    df = load_data("city_temperature.csv")

    # Question 3 - Exploring data for specific country
    israel_subset_df = df[df['Country'] == 'Israel']

    # Convert 'Year' to string for color parameter
    israel_subset_df = israel_subset_df.copy()  # Make a copy to avoid the SettingWithCopyWarning
    israel_subset_df['Year_str'] = israel_subset_df['Year'].astype(str)

    # Plot a scatter plot showing the relation between 'DayOfYear' and 'Temp', color-coded by 'Year'
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(israel_subset_df['DayOfYear'], israel_subset_df['Temp'],
                          c=israel_subset_df['Year_str'].astype('category').cat.codes, cmap='viridis')
    plt.xlabel('Day of Year')
    plt.ylabel('Average Daily Temperature')
    plt.title('Average Daily Temperature as a Function of Day of Year in Israel')

    # Add colorbar with correct labels
    cbar = plt.colorbar(scatter)
    cbar.set_label('Year')
    year_ticks = israel_subset_df['Year_str'].astype('category').cat.categories
    cbar.set_ticks(range(len(year_ticks)))
    cbar.set_ticklabels(year_ticks)

    # plt.show()

    # Save the scatter plot as an image
    plt.savefig('./color_bar_by_year_temperature_vs_dayOfYear.png')

    # Close the scatter plot figure
    plt.close()

    # Group the samples by 'Month' and plot a bar plot showing the standard deviation of the daily temperatures
    monthly_std = israel_subset_df.groupby('Month')['Temp'].agg(['std']).reset_index()

    plt.figure(figsize=(10, 6))
    plt.bar(monthly_std['Month'], monthly_std['std'])
    plt.xlabel('Month')
    plt.ylabel('Standard Deviation of Daily Temperatures')
    plt.title('Standard Deviation of Daily Temperatures by Month in Israel')
    # plt.show()
    plt.savefig('./std_daily_temperatures_by_month_Israel.png')

    plt.close()

    # Question 4 - Exploring differences between countries

    # Group the samples according to 'Country' and 'Month' and calculate the average and standard deviation of the
    # temperature
    monthly_stats = df.groupby(['Country', 'Month'])['Temp'].agg(['mean', 'std']).reset_index()

    # Plot the line plot of the average monthly temperature with error bars color-coded by country
    fig = px.line(monthly_stats, x='Month', y='mean', color='Country', error_y='std',
                  title='Average Monthly Temperature by Country')
    fig.update_layout(
        xaxis_title="Month",
        yaxis_title="Average Temperature"
    )
    fig.write_html('./mean_monthly_temperature_by_country.html')
    # fig.show()

    # Question 5 - Fitting model for different values of `k`

    random_state = np.random.randint(1000)

    # Split the data into training and test sets
    train_df, test_df = train_test_split(israel_subset_df, test_size=0.25, random_state=random_state)

    # Extract features and target
    X_train = train_df['DayOfYear'].values
    y_train = train_df['Temp'].values
    X_test = test_df['DayOfYear'].values
    y_test = test_df['Temp'].values

    # Initialize list to store test errors for different polynomial degrees
    test_errors = []
    degrees = list(range(1, 11))

    # Fit polynomial models and calculate test errors for each degree
    for k in degrees:
        # Initialize PolynomialFitting model
        model = PolynomialFitting(k)
        # Fit model on training data
        model.fit(X_train, y_train)

        # Calculate test error
        test_error = model.loss(X_test, y_test)
        test_errors.append(round(test_error, 2))

    # Print the test errors
    for k, error in enumerate(test_errors, start=1):
        print(f"Test error for polynomial degree {k}: {error}")

    # Plot the test errors with text annotations
    plt.figure(figsize=(10, 6))
    bars = plt.bar(degrees, test_errors)

    # Add text annotations on top of each bar
    for bar, error in zip(bars, test_errors):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 0.05, f'{error}', ha='center', va='bottom')

    plt.xlabel('Polynomial Degree')
    plt.ylabel('Loss')
    plt.title('Test Error for Polynomial Models of Different Degrees')
    plt.xticks(degrees)
    #plt.show()
    plt.savefig("./loss_by_degree_in_Israel.png")
    plt.close()

    # Question 6 - Evaluating fitted model on different countries
    k = 3
    model = PolynomialFitting(k)
    model.fit(israel_subset_df['DayOfYear'].values, israel_subset_df['Temp'].values)
    errors_by_country = {}
    for i, country in enumerate(df['Country'].unique()):
        if country != 'Israel':
            country_subset = df[df['Country'] == country]
            X_country = country_subset['DayOfYear'].values
            y_country = country_subset['Temp'].values
            error = model.loss(X_country, y_country)
            errors_by_country[country] = error

    # Assign unique colors to each country
    colors = plt.cm.tab10(np.linspace(0, 1, len(errors_by_country)))


    # Plot a bar plot showing the model's error over each of the other countries
    plt.figure(figsize=(10, 6))
    plt.bar(list(errors_by_country.keys()), list(errors_by_country.values()), color=colors)
    plt.xlabel('')
    plt.ylabel('Loss')
    plt.title('Model Loss For Each Country (k=3)')
    plt.xticks(rotation=0)

    # Add text annotations for loss values on top of each bar
    for i, country in enumerate(errors_by_country.keys()):
        plt.text(i, errors_by_country[country], f'{errors_by_country[country]:.2f}', ha='center', va='bottom')

    # plt.show()
    plt.savefig("./model_loss_by_country_k=3.png")
    plt.close()

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import statsmodels.api as sm

def preprocess_pollution_data(pollution_df):
    """
    Written for pollution data; will drop null values, reset index, drop 'No' column,
    and rescale 'year' column by subtracting 2010 from each value in that column. Returns
    prepocessed DataFrame.
    """
    raw_num_df_rows = len(pollution_df)
    pollution_df = pollution_df.dropna()
    remaining_num_df_rows = len(pollution_df)
    percent_na = (
        (raw_num_df_rows - remaining_num_df_rows) / raw_num_df_rows * 100
    )
    print(f"Percent of rows dropped: {round(percent_na,2)}%")

    pollution_df.reset_index(drop=True, inplace=True)

    df_preprocessed = pollution_df.drop(columns='No')

    df_preprocessed['year'] = df_preprocessed['year'] - 2010

    return df_preprocessed

def scaler(df):
    """
    Scales the following columns of a preprocessed pollution DataFrame:
    "DEWP", "TEMP", "PRES", "Iws", "Is", "Ir".
    Returns scaled DataFrame.
    """

    columns_to_scale = ["DEWP", "TEMP", "PRES", "Iws", "Is", "Ir"]
    scaled_columns = StandardScaler().fit_transform(df[columns_to_scale])

    df_scaled = pd.DataFrame(scaled_columns, columns=columns_to_scale)
    df_scaled = pd.concat([df[['year', 'month', 'day', 'hour', 'pm2.5', 'cbwd']],
                           df_scaled],
                           axis=1)
    
    return df_scaled

def encoder(df):
    """
    Encodes the 'cbwd' column (wind direction) of a preprocessed and scaled
    pollution DataFrame.
    Returns split into X and y train and test data.
    """

    df_encoded = pd.get_dummies(df, dtype=int)

    # Since 'pm2.5' will be our dependent variable, display correlations between it and the
    # independent variables
    print(f"Correlations between pm2.5 and other features:\n{df_encoded.corr()['pm2.5']}")

    return df_encoded

def select_all_features(df):
    '''
    Define X and y and return train and test data sets
    '''
    X = df.drop(columns=['pm2.5'])
    y = df['pm2.5'].values.reshape(-1, 1)

    return train_test_split(X, y)

def select_reduced_features(df, p_cutoff):
    """
    Selects features based on p-values. Any features with a p-value larger than
    p_cutoff will be dropped.
    Returns X DataFrame with only selected features.
    """

    X = df.drop(columns=['pm2.5'])
    y = df['pm2.5'].values.reshape(-1, 1)

    lr = sm.OLS(y, X).fit()

    p_values = lr.pvalues.sort_values()

    selected_features = p_values.loc[p_values<p_cutoff]
    X_sel = X[selected_features.index]

    return train_test_split(X_sel, y)

def r2_adj(x, y, model):
    """
    Calculates adjusted r-squared values given an X variable, 
    predicted y values, and the model used for the predictions.
    """
    r2 = model.score(x,y)
    n_cols = x.shape[1]
    return 1 - (1 - r2) * (len(y) - 1) / (len(y) - n_cols - 1)

def check_metrics(X_test, y_test, model):
    # Use the pipeline to make predictions
    y_pred = model.predict(X_test)

    # Print out the MSE, r-squared, and adjusted r-squared values
    print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")
    print(f"R-squared: {r2_score(y_test, y_pred)}")
    print(f"Adjusted R-squared: {r2_adj(X_test, y_test, model)}")

    return r2_adj(X_test, y_test, model)

def get_best_pipeline(pipeline_all, pipeline_sel, pollution_df):
    """
    Accepts two pipelines and pollution data.
    Uses two different functions to calculate r2 for data set with all features
    and dataset with select features according to a cutoff p-value.
    Splits the data for training the different 
    pipelines, then evaluates which pipeline performs
    best.
    """
    # Apply the preprocess_pollution_data step
    df_preprocessed = preprocess_pollution_data(pollution_df)

    # Apply scaling
    df_scaled = scaler(df_preprocessed)

    # Encode data
    df_transformed = encoder(df_scaled)

    X_train, X_test, y_train, y_test = select_all_features(df_transformed)

    # Fit the first pipeline
    pipeline_all.fit(X_train, y_train)

    print("Testing all features")
    # Print out the MSE, r-squared, and adjusted r-squared values
    # and collect the adjusted r-squared for the first pipeline
    p_all_adj_r2 = check_metrics(X_test, y_test, pipeline_all)

    # Select features according to p-values larger than p_cutoff
    p_cutoff = 1.0e-75
    X_train, X_test, y_train, y_test = select_reduced_features(df_transformed, p_cutoff)

    # Fit the second pipeline
    pipeline_sel.fit(X_train, y_train)

    print("Testing selected features")
    # Print out the MSE, r-squared, and adjusted r-squared values
    # and collect the adjusted r-squared for the second pipeline
    p_sel_adj_r2 = check_metrics(X_test, y_test, pipeline_sel)

    # Compare the adjusted r-squared for each pipeline and 
    # return the best model
    if p_sel_adj_r2 > p_all_adj_r2:
        print("Returning selected feature pipeline")
        return pipeline_sel
    else:
        print("Returning all feature pipeline")
        return pipeline_all

def pollutiuon_model_generator(pollution_df):
    """
    Defines a series of steps that will preprocess data,
    split data, and train a model for predicting rent prices
    using linear regression. It will return the trained model
    and print the mean squared error, r-squared, and adjusted
    r-squared scores.
    """
    # Create a list of steps for a pipeline that will one hot encode and scale data
    # Each step should be a tuple with a name and a function
    steps = [("Linear Regression", LinearRegression())] 

    # Create a pipeline object
    pipeline = Pipeline(steps)

    # Create a second pipeline object
    pipeline2 = Pipeline(steps)

    # Get the best pipeline
    pipeline = get_best_pipeline(pipeline, pipeline2, pollution_df)

    # Return the trained model
    return pipeline

if __name__ == "__main__":
    print("This script should not be run directly! Import these functions for use in another file.")
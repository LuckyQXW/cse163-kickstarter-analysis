import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
sns.set()


def preprocess_data(data):
    """
    Takes in a Pandas DataFrame containing the data from the Kickstarter
    platform. Filters down to only include projects with successful and
    failed states. Calculates the pledged ratio for each project, extracts
    the launched month from the launched date index, calculates the duration
    of each campaign, and stores them in new columns.
    Returns a new DataFrame with the extra information.
    """
    successful = data['state'] == 'successful'
    failed = data['state'] == 'failed'
    data = data[successful | failed]
    data['pledged_ratio'] = data['usd_pledged_real'] / data['usd_goal_real']
    data['launched'] = data.index
    data['launched_month'] = data['launched'].apply(get_month)
    data['launched'] = pd.to_datetime(data['launched'])
    data['deadline'] = pd.to_datetime(data['deadline'])
    data['duration'] = (data['deadline'] - data['launched']) \
        .apply(get_duration_in_days)
    return data


def get_month(date):
    """
    Takes in a DatetimeIndex and returns the month as a String.
    """
    return (str)(date.month)


def get_duration_in_days(timedelta):
    """
    Takes in a TimeDelta and returns the number of days in the TimeDelta
    object as a numpy.int64.
    """
    return timedelta.days


def classifier(data, features, label, goal_max, goal_min=250, depth=30):
    """
    Takes in a Pandas DataFrame containing the data from the Kickstarter
    platform, a list of feature column names features, a String label, the
    maximum usd goal amount goal_max (exclusive), the minimum usd goal amount
    goal_min (inclusive), and the max depth of the DecisionTreeClassifier.
    If the goal_min is not not set, it is defaulted to 250 to avoid outliers
    with very small goal amounts. If the depth is not set, it is defaulted to
    30. Trains the input data in a DecisionTreeClassifier.
    Returns the result in the a tuple with the first element be the accuracy
    score of the test set and the second element be a list of feature
    importances sorted in descending order in the form of
    (feature, importance).
    """
    data = preprocess_data(data)
    filtered = data[(data['usd_goal_real'] < goal_max)
                    & (data['usd_goal_real'] >= goal_min)]
    features.append(label)
    filtered_ml = filtered[features]
    X = filtered_ml.loc[:, filtered_ml.columns != label]
    X = pd.get_dummies(X)
    y = filtered_ml[label]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = DecisionTreeClassifier(max_depth=depth)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    importances = list(zip(X.columns, model.feature_importances_))
    importances = sorted(importances, key=lambda x: x[1], reverse=True)

    return accuracy_score(y_test, y_pred), importances


def regressor(data, features, label, goal_max, goal_min=250):
    """
    Takes in a Pandas DataFrame containing the data from the Kickstarter
    platform, a list of feature column names features, a String label, the
    maximum usd goal amount goal_max (exclusive), the minimum usd goal amount
    goal_min (inclusive).
    If the goal_min is not not set, it is defaulted to 250 to avoid outliers
    with very small goal amounts.
    Returns the result in the a tuple with the first element be the mean
    squared error of the test set and the second element be a list of feature
    importances sorted in descending order in the form of
    (feature, importance).
    """
    data = preprocess_data(data)
    filtered = data[(data['usd_goal_real'] < goal_max)
                    & (data['usd_goal_real'] >= goal_min)]
    features.append(label)
    filtered_ml = filtered[features]
    X = filtered_ml.loc[:, filtered_ml.columns != label]
    X = pd.get_dummies(X)
    y = filtered_ml[label]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = DecisionTreeRegressor()
    model.fit(X_train, y_test)
    y_pred = model.predict(X_test)

    importances = list(zip(X.columns, model.feature_importances_))
    importances = sorted(importances, key=lambda x: x[1], reverse=True)

    return mean_squared_error(y_test, y_pred), importances


def graph_optimal_depth(data, features, label, goal_max=5000, goal_min=250,
                        max_depth=20):
    """
    Takes in a Pandas DataFrame containing the data from the Kickstarter
    platform, a list of feature column names features, a String label, the
    maximum usd goal amount goal_max (exclusive), the minimum usd goal amount
    goal_min (inclusive), and the max_depth of the DecisionTreeClassifier
    If the goal_min is not not set, it is defaulted to 250 to avoid outliers
    with very small goal amounts. If the goal_max is not set, it is defaulted
    to 5000. If the max_depth is not set, it is defaulted to 20.
    Generates a line plot showing the test accuracy with respect to the max
    depth for visualizing the optimal max_depth for the classifier.
    """
    graph_data = []
    for i in range(1, max_depth):
        score, importance = classifier(data, features, label, goal_max,
                                       goal_min, i)
        graph_data.append({'max depth': i, 'test accuracy': score})
    graph_data = pd.DataFrame(graph_data)
    sns.relplot(kind='line', x='max depth', y='test accuracy', data=graph_data)

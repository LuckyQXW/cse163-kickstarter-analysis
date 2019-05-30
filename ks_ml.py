import cse163_utils  # noqa: F401
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
import main
sns.set()


def classifier(data, features, feature_index, label, max_goal, min_goal=250):
    """
    Takes in a Pandas DataFrame containing the data from the Kickstarter
    platform, a list of feature column names features, an integer feature
    index for differentiating different feature sets, a String label, the
    maximum usd goal amount max_goal (exclusive), and the minimum usd goal
    amount min_goal (inclusive). If the min_goal is not not set, it is
    defaulted to 250.
    Saves a graph showing max depth vs accuracy with the file name
    max_depth_vs_accuracy_max_<max_goal>(<feature_index>).jpg
    Returns the result in the a tuple with the first element be the accuracy
    score of the test set and the second element be a list of feature
    importances sorted in descending order in the form of
    (feature, importance).
    """
    filtered = data[(data['usd_goal_real'] < max_goal)
                    & (data['usd_goal_real'] >= min_goal)]
    columns = features.copy()
    columns.append(label)
    filtered_ml = filtered[columns]
    X = filtered_ml.loc[:, filtered_ml.columns != label]
    X = pd.get_dummies(X)
    y = filtered_ml[label]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # Find the best depth using 7-fold cross validation and graph the result
    depth = []
    for i in range(3, 20):
        clf = DecisionTreeClassifier(max_depth=i)
        scores = cross_val_score(estimator=clf, X=X_train, y=y_train, cv=7,
                                 n_jobs=4)
        depth.append({'Max Depth': i, 'Score': scores.mean()})
    graph_data = pd.DataFrame(depth)
    graph_optimal_depth('max_depth_vs_accuracy_max_' + str(max_goal)
                        + '(' + str(feature_index) + ')' + '.jpg',
                        graph_data, 'Max Depth', 'Score',
                        features, max_goal)
    # Make prediction on test set using the best depth
    best_depth = graph_data.nlargest(1, 'Score')['Max Depth'].iloc[0]
    clf = DecisionTreeClassifier(max_depth=best_depth)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    importances = list(zip(X_train.columns, clf.feature_importances_))
    importances = sorted(importances, key=lambda x: x[1], reverse=True)
    return accuracy_score(y_pred, y_test), importances


def regressor(data, features, feature_index, label, max_goal, min_goal=250):
    """
    Takes in a Pandas DataFrame containing the data from the Kickstarter
    platform, a list of feature column names features, an integer feature
    index for differentiating different feature sets, a String label, the
    maximum usd goal amount max_goal (exclusive), and the minimum usd goal
    amount min_goal (inclusive). If the min_goal is not not set, it is
    defaulted to 250.
    Saves a graph showing max depth vs accuracy with the file name
    max_depth_vs_accuracy_max_<max_goal>(<feature_index>).jpg
    Returns the result in the a tuple with the first element be the accuracy
    score of the test set and the second element be a list of feature
    importances sorted in descending order in the form of
    (feature, importance).
    """
    filtered = data[(data['usd_goal_real'] < max_goal)
                    & (data['usd_goal_real'] >= min_goal)]
    columns = features.copy()
    columns.append(label)
    filtered_ml = filtered[columns]
    X = filtered_ml.loc[:, filtered_ml.columns != label]
    X = pd.get_dummies(X)
    y = filtered_ml[label]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # Find the best depth using 7-fold cross validation and graph the result
    depth = []
    for i in range(3, 20):
        clf = DecisionTreeRegressor(max_depth=i)
        scores = cross_val_score(estimator=clf, X=X_train, y=y_train, cv=7,
                                 n_jobs=4, scoring='neg_mean_squared_error')
        depth.append({'Max Depth': i,
                      'Negative Mean Square Error': scores.mean()})
    graph_data = pd.DataFrame(depth)
    graph_optimal_depth('max_depth_vs_mean_squared_error_max_' + str(max_goal)
                        + '(' + str(feature_index) + ')' + '.jpg', graph_data,
                        'Max Depth', 'Negative Mean Square Error',
                        features, max_goal)
    # Make prediction on test set using the best depth
    best_depth = graph_data. \
        nlargest(1, 'Negative Mean Square Error')['Max Depth'].iloc[0]
    clf = DecisionTreeRegressor(max_depth=best_depth)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    importances = list(zip(X_train.columns, clf.feature_importances_))
    importances = sorted(importances, key=lambda x: x[1], reverse=True)
    return mean_squared_error(y_pred, y_test), importances


def graph_optimal_depth(filename, data, x, y, features, max_goal,
                        min_goal=250):
    """
    Takes in a String filename for the saved graph, a Pandas DataFrame
    containing the data with each max_depth and corresponding measure, name of
    the column for x axis, name of the column for y axis, a list of
    feature column names features, the maximum usd goal amount
    max_goal (exclusive), the minimum usd goal amount min_goal (inclusive),
    and the max_depth of the DecisionTreeClassifier.
    If the min_goal is not not set, it is defaulted to 250 to avoid outliers
    with very small goal amounts.
    Generates a line plot showing the test score with respect to the max
    depth for visualizing the optimal max_depth for the classifier.
    """
    data.plot(x=x, y=y, legend=False)
    plt.suptitle("Max Goal: $" + str(max_goal) + "\n" +
                 "Features: " + ", ".join(str(x) for x in features),
                 fontsize=12)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.savefig(filename)
    plt.clf()


def print_result(measure, value, importances):
    """
    Takes in a String measure describing the type of the value used to evaluate
    a machine learning model, a float value representing the accuracy/error
    value depends on the machine learning model, and a list of tuples in the
    form of (feature, importance), prints the given information in a formatted
    manner.
    """
    print(measure + ": " + str(value))
    print("Feature importance ranking: ")
    for f in importances:
        print(f)
    print()


def run():
    data = main.preprocess_data('ks-projects-201801.csv')
    successful = data['state'] == 'successful'
    failed = data['state'] == 'failed'
    data = data[successful | failed]
    label = 'state'
    label2 = 'pledged_ratio'

    # First feature combo with backers
    features1 = ['usd_goal_real', 'backers', 'launched_month', 'main_category',
                 'duration']
    # Second feature combo without backers, and usd_goal_real
    features2 = ['launched_month', 'main_category', 'duration']

    # Prints the accuracy and feature importance ranking using the best depth
    # for the DecisionTreeClassifier
    print("Result for classifying success/fail with feature set 1, " +
          "max goal $10000: ")
    accuracy, importances = classifier(data, features1, 1, label, 10000)
    print_result("Accuracy Score", accuracy, importances)

    print("Result for classifying success/fail with feature set 2, " +
          "max goal $10000: ")
    accuracy, importances = classifier(data, features2, 2, label, 10000)
    print_result("Accuracy Score", accuracy, importances)

    print("Result for classifying success/fail with feature set 1, " +
          "max goal $20000: ")
    accuracy, importances = classifier(data, features1, 1, label, 20000)
    print_result("Accuracy Score", accuracy, importances)

    print("Result for classifying success/fail with feature set 2, " +
          "max goal $20000: ")
    accuracy, importances = classifier(data, features2, 2, label, 20000)
    print_result("Accuracy Score", accuracy, importances)

    # Prints the accuracy and feature importance ranking using the best depth
    # for the DecisionTreeRegressor
    print("Result for predicting pledged ratio with feature set 1, " +
          "max goal $10000: ")
    error, importances = regressor(data, features1, 1, label2, 10000)
    print_result("Mean Squared Error", error, importances)

    print("Result for predicting pledged ratio with feature set 2, " +
          "max goal $10000: ")
    error, importances = regressor(data, features2, 2, label2, 10000)
    print_result("Mean Squared Error", error, importances)

    print("Result for predicting pledged ratio with feature set 1, " +
          "max goal $20000: ")
    error, importances = regressor(data, features1, 1, label2, 20000)
    print_result("Mean Squared Error", error, importances)

    print("Result for predicting pledged ratio with feature set 2, " +
          "max goal $20000: ")
    error, importances = regressor(data, features2, 2, label2, 20000)
    print_result("Mean Squared Error", error, importances)


if __name__ == '__main__':
    run()

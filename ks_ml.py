import cse163_utils  # noqa: F401
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
import main
sns.set()


def classifier(data, features, label, max_goal, min_goal=250, depth=5):
    """
    Takes in a Pandas DataFrame containing the data from the Kickstarter
    platform, a list of feature column names features, a String label, the
    maximum usd goal amount max_goal (exclusive), the minimum usd goal amount
    min_goal (inclusive), and the max depth of the DecisionTreeClassifier.
    If the min_goal is not not set, it is defaulted to 250. If the depth is not
    set, it is defaulted to 5. Trains the input data in a
    DecisionTreeClassifier.
    Returns the result in the a tuple with the first element be the accuracy
    score of the test set and the second element be a list of feature
    importances sorted in descending order in the form of
    (feature, importance).
    """
    X_train, X_test, y_train, y_test = \
        get_splited_data(data, features, label, max_goal, min_goal)
    model = DecisionTreeClassifier(max_depth=depth, random_state=1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    importances = list(zip(X_train.columns, model.feature_importances_))
    importances = sorted(importances, key=lambda x: x[1], reverse=True)

    return accuracy_score(y_test, y_pred), importances


def regressor(data, features, label, max_goal, min_goal=250):
    """
    Takes in a Pandas DataFrame containing the data from the Kickstarter
    platform, a list of feature column names features, a String label, the
    maximum usd goal amount max_goal (exclusive), the minimum usd goal amount
    min_goal (inclusive).
    If the min_goal is not not set, it is defaulted to 250.
    Returns the result in the a tuple with the first element be the mean
    squared error of the test set and the second element be a list of feature
    importances sorted in descending order in the form of
    (feature, importance).
    """
    X_train, X_test, y_train, y_test = get_splited_data(data, features, label,
                                                        max_goal, min_goal)
    model = DecisionTreeRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    importances = list(zip(X_train.columns, model.feature_importances_))
    importances = sorted(importances, key=lambda x: x[1], reverse=True)

    return mean_squared_error(y_test, y_pred), importances


def get_splited_data(data, features, label, max_goal, min_goal=250):
    """
    Takes in a Pandas DataFrame containing the data from the Kickstarter
    platform, a list of feature column names features, a String label, the
    maximum usd goal amount max_goal (exclusive), the minimum usd goal amount
    min_goal (inclusive).
    If the min_goal is not not set, it is defaulted to 250.
    Returns the splited data in the form of (X_train, X_test, y_train, y_test)
    """
    filtered = data[(data['usd_goal_real'] < max_goal)
                    & (data['usd_goal_real'] >= min_goal)]
    columns = features.copy()
    columns.append(label)
    filtered_ml = filtered[columns]
    X = filtered_ml.loc[:, filtered_ml.columns != label]
    X = pd.get_dummies(X)
    y = filtered_ml[label]
    return train_test_split(X, y, test_size=0.2, random_state=3)


def graph_optimal_depth(filename, data, features, label, max_goal=5000,
                        min_goal=250, max_depth=20):
    """
    Takes in a String filename for the saved graph, a Pandas DataFrame
    containing the data from the Kickstarter platform, a list of feature
    column names features, a String label, the maximum usd goal amount
    max_goal (exclusive), the minimum usd goal amount
    min_goal (inclusive), and the max_depth of the DecisionTreeClassifier
    If the min_goal is not not set, it is defaulted to 250 to avoid outliers
    with very small goal amounts. If the max_goal is not set, it is defaulted
    to 5000. If the max_depth is not set, it is defaulted to 20.
    Generates a line plot showing the test accuracy with respect to the max
    depth for visualizing the optimal max_depth for the classifier.
    """
    graph_data = []
    label = 'state'
    for i in range(2, max_depth):
        score, importance = \
            classifier(data, features, label, max_goal, min_goal, i)
        graph_data.append({'max depth': i, 'test accuracy': score})
    graph_data = pd.DataFrame(graph_data)
    graph_data.plot(x='max depth', y='test accuracy')
    plt.suptitle("Max Goal: $" + str(max_goal) + "\n" +
                 "Features: " + ", ".join(str(x) for x in features),
                 fontsize=12)
    plt.savefig(filename)


def graph_classification_combo(data, features, feature_index, label, max_goal):
    """
    Takes in a Pandas DataFrame containing the data from the Kickstarter
    platform, a list of feature column names features, a integer representing
    the unique index of a given set of features, a String label,
    the maximum usd goal amount max_goal (exclusive).
    Generates a line plot showing the test accuracy with respect to the max
    depth for visualizing the optimal max_depth for the classifier with the
    given feature and label combination.
    """
    graph_optimal_depth('max_depth_vs_accuracy_max_' + str(max_goal)
                        + '(' + str(feature_index) + ')' + '.jpg',
                        data, features, label, max_goal)


def print_result(measure, value, importances):
    """
    Takes in a String measure describing the type of the value used to evaluate
    a machine learning model, a float value representing the accuracy/error
    value depends on the machine learning model, and a list of tuples in the
    form of (feature, importance), prints the given information in a formatted
    manner.
    """
    print(measure + ": " + str(value))
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

    # Graphs best depth using different combos
    graph_classification_combo(data, features1, 1, label, 10000)
    plt.clf()
    graph_classification_combo(data, features1, 1, label, 20000)
    plt.clf()
    graph_classification_combo(data, features2, 2, label, 10000)
    plt.clf()
    graph_classification_combo(data, features2, 2, label, 20000)
    plt.clf()

    # Prints the accuracy and feature importance ranking using the best depth
    # for the DecisionTreeClassifier
    accuracy, importances = classifier(data, features1, label, 10000, depth=9)
    print_result("Accuracy Score", accuracy, importances)
    accuracy, importances = classifier(data, features2, label, 10000, depth=12)
    print_result("Accuracy Score", accuracy, importances)

    # Prints the accuracy and feature importance ranking using regressor
    error, importances = regressor(data, features1, label2, 10000)
    print_result("Mean Square Error", error, importances)
    error, importances = regressor(data, features2, label2, 10000)
    print_result("Mean Square Error", error, importances)


if __name__ == '__main__':
    run()

'''
This file contains functions that analyze Kickstarter data with
DecisionTreeClassifiers.
'''
import cse163_utils  # noqa: F401
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.tree import export_graphviz
import graphviz


def classifier(data, features, output_index, label, max_goal, min_goal=250,
               test=False, graph=False):
    '''
    Takes in a Pandas DataFrame containing the data from the Kickstarter
    platform, a list of feature column names features, an integer feature
    index for differentiating different output files, a String label, the
    maximum usd goal amount max_goal (exclusive), and the minimum usd goal
    amount min_goal (inclusive). If the min_goal is not not set, it is
    defaulted to 250.
    By default, test and graph are set to False.
    If test is True, saves a graph showing max depth vs accuracy with the
    filename max_depth_vs_accuracy_<min_goal>_to_<max_goal>(<output_index>).jpg
    in the test folder.
    If graph is True, saves the visualization of the decision tree model in
    model.gv in the results folder.
    Returns the result in the a tuple with the first element be the accuracy
    score of the test set and the second element be a list of feature
    importances sorted in descending order in the form of
    (feature, importance).
    '''
    X_train, X_test, y_train, y_test = \
        get_splitted_data(data, features, label, max_goal, min_goal)
    # Find the best depth using 5-fold cross validation and graph the result
    depth = []
    for i in range(3, 20):
        clf = DecisionTreeClassifier(max_depth=i)
        scores = cross_val_score(estimator=clf, X=X_train, y=y_train, cv=5,
                                 n_jobs=2)
        depth.append({'Max Depth': i, 'Score': scores.mean()})
    graph_data = pd.DataFrame(depth)
    # Save max depth vs accuracy graph to test folder if test=True
    if test:
        graph_optimal_depth('max_depth_vs_accuracy_goal_' + str(min_goal)
                            + '_to_' + str(max_goal)
                            + '(' + str(output_index) + ')' + '.jpg',
                            graph_data, 'Max Depth', 'Score',
                            features, max_goal)
    # Make prediction on test set using the best depth
    best_depth = graph_data.nlargest(1, 'Score')['Max Depth'].iloc[0]
    print('Predicting test set using the depth of: ' + str(best_depth))
    clf = DecisionTreeClassifier(max_depth=best_depth)
    clf.fit(X_train, y_train)
    # Save source file of tree visualization to results folder if graph=True
    if graph:
        save_graphviz_source(clf, X_train, y_train)
    y_pred = clf.predict(X_test)
    importances = list(zip(X_train.columns, clf.feature_importances_))
    importances = sorted(importances, key=lambda x: x[1], reverse=True)
    return accuracy_score(y_pred, y_test), importances


def get_splitted_data(data, features, label, max_goal, min_goal):
    '''
    Takes in a Pandas DataFrame containing the data from the Kickstarter
    platform, a list of feature column names features, an integer feature
    index for differentiating different feature sets, a String label, the
    maximum usd goal amount max_goal (exclusive), and the minimum usd goal
    amount min_goal (inclusive).
    Returns the data splitted in the training set and the test set in the
    form of X_train, X_test, y_train, y_test.
    '''
    filtered = data[(data['usd_goal_real'] < max_goal)
                    & (data['usd_goal_real'] >= min_goal)]
    columns = features.copy()
    columns.append(label)
    filtered_ml = filtered[columns]
    X = filtered_ml.loc[:, filtered_ml.columns != label]
    X = pd.get_dummies(X)
    y = filtered_ml[label]
    return train_test_split(X, y, test_size=0.2)


def graph_optimal_depth(filename, data, x, y, features, max_goal,
                        min_goal=250):
    '''
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
    '''
    data.plot(x=x, y=y, legend=False)
    plt.suptitle('Max Goal: $' + str(max_goal) + '\n' +
                 'Features: ' + ', '.join(str(x) for x in features),
                 fontsize=12)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.savefig('test/' + filename)
    plt.clf()


def save_graphviz_source(clf, X, y):
    '''
    Takes in a DecisionTreeClassifier model clf, a list of features used to
    train the model X, and the corresponding label y, save a graphviz source
    file in the results folder for visualization.
    '''
    dot_data = export_graphviz(clf,
                               feature_names=X.columns,
                               class_names=y.unique(),
                               filled=True, rounded=False,
                               special_characters=True)
    graphviz.Source(dot_data).save(filename='results/model.gv')


def classifier_trial(data, features, output_index, label, max_goal,
                     min_goal=250, test=False, graph=False):
    '''
    Runs a trial with DecisionTreeClassifier with the given data, features,
    output_index, label, and max_goal amount, and min_goal amount.
    If min_goal is not set, it is defaulted to 250. Can also get test
    depth tuning visualization by setting test to True or visualization of the
    decision tree by setting graph to True.
    Writes the resulting accuracy score on the test set and also the
    feature importance ranking to a txt file.
    '''
    f = open('results/ml_output' + str(output_index) + '.txt', mode='a',
             encoding='utf-8')
    f.write('Result for classifying success/fail with feature set ' +
            str(output_index) +
            ', goal $' + str(min_goal) + '~$' + str(max_goal) + ': \n')
    accuracy, importances = \
        classifier(data, features, output_index, label, max_goal, min_goal,
                   test=test, graph=graph)
    f.write('Accuracy Score:' + str(accuracy) + '\n')
    f.write('Feature importance ranking: \n')
    for feature in importances:
        f.write(feature[0] + ' ' + str(feature[1]) + '\n')
    f.write('\n')
    f.close()


def resolve_and_percent(category):
    '''
    Extra file processing changing & to and for Film & Video category for
    graphviz dot file parsing.
    '''
    if category == 'Film & Video':
        return 'Film and Video'
    return category


def run(data):
    '''
    Takes in preprocessed Kickstarter data and perform machine learning
    analysis.
    '''
    successful = data['state'] == 'successful'
    failed = data['state'] == 'failed'
    data.loc[:, 'main_category'] = \
        data['main_category'].apply(resolve_and_percent)
    data = data[successful | failed]
    label = 'state'

    # First feature combo with backers
    features1 = ['usd_goal_real', 'backers', 'launched_month', 'main_category',
                 'duration']
    # Second feature combo without backers
    features2 = ['usd_goal_real', 'launched_month', 'main_category',
                 'duration']

    # Prints the accuracy and feature importance ranking using the best depth
    # for the DecisionTreeClassifier with varied max goal amount
    # Trials with feature set 1
    classifier_trial(data, features1, 1, label, max_goal=10000, test=True)
    classifier_trial(data, features1, 1, label, max_goal=20000)
    classifier_trial(data, features1, 1, label, max_goal=30000)
    classifier_trial(data, features1, 1, label, max_goal=40000)
    # Trials with feature set 2
    classifier_trial(data, features2, 2, label, max_goal=10000)
    classifier_trial(data, features2, 2, label, max_goal=20000)
    classifier_trial(data, features2, 2, label, max_goal=30000)
    classifier_trial(data, features2, 2, label, max_goal=40000)
    # Trials with smaller goal range
    classifier_trial(data, features1, 3, label,
                     max_goal=5000, min_goal=3000, graph=True)

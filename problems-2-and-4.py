# Problems 2 and 4
# Paul Pham pkdpham
#
# This program defines functions that will work with the Kickstarter dataset to
# do various forms of statistical analysis

import math
from main import preprocess_data as preprocess


def cut_and_arrange(data, percent):
    '''
    Arranges the given data in descending order by number of backers and
    returns a subset of the data based on the given percentile of the top
    rows from the dataset
    '''
    sorted = data.sort_values(by='backers', ascending=False)
    percentile = int(math.floor(percent * sorted.shape[0]))
    subset = sorted.iloc[:percentile, ]
    return subset


def get_unique(data, feature):
    '''
    Calculates and returns all unique values in a given column as a list
    '''
    uniq_col_values = list(data[feature].unique())
    return uniq_col_values


def sample_statistics(data, categories, feature):
    '''
    Calculates the percentages of each given feature and returns them
    mapped to their feature in a dictionary
    '''
    result = dict.fromkeys(categories, 0)
    for f in data[feature]:
        result[f] += 1
    for category in result:
        result[category] = round((result[category] / len(data)) * 100, 2)
    return result


def main():
    data = preprocess('ks-projects-201801.csv')
    subset = cut_and_arrange(data, .10)
    unique_categories = get_unique(subset, 'main_category')
    percentages = sample_statistics(subset, unique_categories, 'main_category')
    print(percentages)
    # TODO: visualize percentages (pie chart? bar chart?)


if __name__ == '__main__':
    main()

# Problems 2 and 4
# Paul Pham pkdpham
#
# This program defines functions that will work with the Kickstarter dataset to
# do various forms of statistical analysis

import math
from main import preprocess_data as preprocess
import matplotlib.pyplot as plt


def cut_and_arrange(data, percent):
    '''
    Arranges the given data in descending order by number of backers and
    returns a subset of the data based on the given percentile of the top
    rows from the dataset.
    Problem 2
    '''
    sorted = data.sort_values(by='backers', ascending=False)
    percentile = int(math.floor(percent * sorted.shape[0]))
    subset = sorted.iloc[:percentile, ]
    return subset


def get_unique(data, feature):
    '''
    Calculates and returns all unique values in a given column as a list.
    Problem 2
    '''
    uniq_col_values = list(data[feature].unique())
    return uniq_col_values


def sample_statistics(data, categories, feature):
    '''
    Calculates the percentages of each given feature and returns them
    mapped to their feature in a dictionary.
    Problem 2
    '''
    result = dict.fromkeys(categories, 0)
    for f in data[feature]:
        result[f] += 1
    for category in result:
        result[category] = round((result[category] / len(data)) * 100, 2)
    return result


def success_per_country(data, state):
    '''
    Returns a dataframe with the rates of the given project state per country
    '''
    successes = data['state'] == state
    total_projects = data.groupby('country')['state'].count()
    of_state = data[successes].groupby('country')['state'].count()
    of_state = round((of_state / total_projects) * 100, 2)
    return of_state


def main():
    data = preprocess('ks-projects-201801.csv')

    # .10 in this case indicates the top 10% of the data
    subset = cut_and_arrange(data, .10)
    unique_categories = get_unique(subset, 'main_category')
    percentages = sample_statistics(subset, unique_categories, 'main_category')

    # for output confirmation purposes only
    print(unique_categories)
    print(percentages)

    success_rates = success_per_country(data, 'successful').to_frame()

    # TODO: visualize percentages (pie chart? bar chart?)
    success_rates.plot(kind='bar', legend=False)
    plt.title('Kickstarter Project Success Rates Per Country')
    plt.xlabel('Countries')
    plt.ylabel('Percent Projects Successful')
    plt.savefig('success_rates.jpg')
    plt.show()


if __name__ == '__main__':
    main()

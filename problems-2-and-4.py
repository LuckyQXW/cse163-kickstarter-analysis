# Problems 2 and 4
# Paul Pham pkdpham
#
# This program defines functions that will work with the Kickstarter dataset to
# do various forms of statistical analysis


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
    percentile = round(percent * sorted.shape[0])
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
    Problem 4
    '''
    successes = data['state'] == state
    total_projects = data.groupby('country')['state'].count()
    of_state = data[successes].groupby('country')['state'].count()
    of_state = round((of_state / total_projects) * 100, 2)
    return of_state


def graph(data, percentile):
    title = percentile + ' Percentile of the Data'
    labels = list(data.keys())

    plt.pie(data.values(), labels=labels)
    plt.title(title)

    plt.show()
    # return plot


def main():
    data = preprocess('ks-projects-201801.csv')

    # .10 in this case indicates the top 10% of the data
    # fifth = cut_and_arrange(data, 0.05)
    tenth = cut_and_arrange(data, 0.10)
    # twenty_fifth = cut_and_arrange(data, 0.25)
    unique_categories = get_unique(tenth, 'main_category')
    percentages = sample_statistics(tenth, unique_categories, 'main_category')

    graph(percentages, "Tenth")
    '''
    success_rates = success_per_country(data, 'successful').to_frame()
    success_rates = success_rates.sort_values('state', ascending=False)

    # TODO: visualize percentages (pie chart? bar chart?)
    success_rates.plot(kind='bar', legend=False, figsize=(12, 9))
    plt.rcParams['font.size'] = 25
    plt.title('Kickstarter Project Success Rates Per Country')
    plt.xlabel('Countries')
    plt.ylabel('Percent Projects Successful')
    plt.savefig('success_rates.jpg')
    '''
    # write results to an output file
    # create and save visualizations for summary statistics (nested pie)


if __name__ == '__main__':
    main()

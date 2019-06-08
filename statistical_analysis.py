# Problems 2 and 4
# Paul Pham pkdpham
#
# This program defines functions that will work with the Kickstarter
# dataset to do various forms of statistical analysis


from main import preprocess_data as preprocess
import matplotlib.pyplot as plt


def cut_and_arrange(data, percent):
    '''
    Arranges the given data in descending order by number of backers and
    returns a subset of the data based on the given percentile of the top
    rows from the dataset.
    '''
    sorted = data.sort_values(by='backers', ascending=False)
    percentile = round(percent * sorted.shape[0])
    subset = sorted.iloc[:percentile, ]
    return subset


def get_unique(data, feature):
    '''
    Calculates and returns all unique values in a given column as a list.
    '''
    uniq_col_values = list(data[feature].unique())
    return uniq_col_values


def get_percentages(data, categories, feature):
    '''
    Calculates the percentages of each given feature and returns them
    mapped to their feature in a dictionary.
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


def graph_perc(data, percentile, pathname, test=False):
    '''
    Takes the given data and graphs a pie chart with labels based on
    the given percentile, and saves it to a results folder with a name
    including the given path
    '''
    colors = ['#C0392B', '#E74C3C', '#9B59B6', '#8E44AD', '#2980B9',
              '#3498DB', '#1ABC9C', '#16A085', '#27AE60', '#2ECC71',
              '#F1C40F', '#F39C12', '#E67E22', '#D35400', '#CD6155']
    title = 'Categories of the ' + percentile + ' Percentile of the Data'
    labels = list(data.keys())
    explode = (0, 0, 0, 0, 0, 0, 0.1, 0, 0, 0, 0, 0, 0, 0, 0)
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.rcParams['font.size'] = 15
    ax.pie(data.values(), labels=labels, colors=colors, startangle=210,
           explode=explode)
    plt.title(title)
    if test:
        plt.savefig('test/test_percentages_graph.jpg')
    else:
        path = 'results/' + pathname + '_perc_categories.jpg'
        plt.savefig(path)


def graph_success(data):
    '''
    Arranges data into descending order and graphs a bar chart depicting
    the success rates of Kickstarter projects per country, saving it to
    a results folder.
    '''
    data = data.sort_values('state', ascending=False)
    plt.rcParams['font.size'] = 25
    data.plot(kind='bar', legend=False, figsize=(15, 15))
    plt.title('Kickstarter Project Success Rates Per Country')
    plt.xlabel('Countries')
    plt.ylabel('Percent Projects Successful')
    plt.savefig('results/success_rates_per_country.jpg')


def main():
    '''
    data = preprocess('ks-projects-201801.csv')

    # cut and store the data based on percentiles
    first = cut_and_arrange(data, .01)
    fifth = cut_and_arrange(data, 0.05)
    tenth = cut_and_arrange(data, 0.10)

    # get unique categories
    categories = get_unique(data, 'main_category')

    # calculate statistics on each percentile
    first = get_percentages(first, categories, 'main_category')
    fifth = get_percentages(fifth, categories, 'main_category')
    tenth = get_percentages(tenth, categories, 'main_category')

    # graph each percentile
    graph_perc(first, '1st', 'first')
    graph_perc(fifth, '5th', 'fifth')
    graph_perc(tenth, '10th', 'tenth')

    # calculate and graph success rates per country
    success_rates = success_per_country(data, 'successful').to_frame()
    graph_success(success_rates)
    '''
    data = preprocess('test2-dataset.csv')
    print(type(success_per_country(data, 'successful')))


if __name__ == '__main__':
    main()

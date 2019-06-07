'''
This file contains functions for analyzing launched time for all Kickstarter
projects.
'''
import cse163_utils  # noqa: F401
import matplotlib.pyplot as plt
import main


def get_path(test):
    '''
    Takes in a boolean test and determines the file path for saving figures.
    Returns 'results/' if false, and 'test/' if true.
    '''
    if test:
        return 'test/'
    return 'results/'


def plot_success_fail_vs_total(data, test=False):
    '''
    Takes in a Pandas DataFrame containing the Kickstarter data and plot the
    number of total, successful, and failed projects over time.
    '''
    data_success = data[data['state'] == 'successful']
    data_fail = data[data['state'] == 'failed']
    data_count = data.resample('M').count()
    data_success_count = data_success.resample('M').count()
    data_fail_count = data_fail.resample('M').count()
    fig, ax = plt.subplots(1)
    data_count['ID'].plot(ax=ax, label='Total', legend=True)
    data_success_count['ID'].plot(ax=ax, label='Successful', legend=True)
    data_fail_count['ID'].plot(ax=ax, label='Failed', legend=True)
    plt.title('Number of Total, Successful, and Failed projects'
              + ' from 2010 to 2017')
    plt.xlabel('Launched Time (Year)')
    plt.ylabel('Count')
    plt.savefig(get_path(test) + 'project_count_over_time.jpg')
    plt.clf()


def plot_success_rate(data, test=False):
    '''
    Takes in a Pandas DataFrame containing the Kickstarter data and plot the
    percentage of successful projects over time.
    '''
    data_success = data[data['state'] == 'successful']
    data_success_count = data_success.resample('M').count()
    data_count = data.resample('M').count()
    data_success_rate = data_success_count / data_count * 100
    data_success_rate['ID'].plot()
    plt.title('Project Success Rate from 2010 to 2017')
    plt.xlabel('Launched Time (Year)')
    plt.ylabel('% of Successful Projects')
    plt.savefig(get_path(test) + 'success_rate_over_time.jpg')
    plt.clf()


def plot_month_counts(data, test=False):
    '''
    Takes in a Pandas DataFrame containing the Kickstarter data and plot the
    total number of project launched with respect to each month.
    '''
    data.loc[:, 'launched_month'] = \
        data.loc[:, 'launched_month'].apply(lambda x: int(x))
    data.groupby('launched_month').count()['ID'].plot.bar()
    plt.title('Total Number of Projects Launched Each Month')
    plt.xlabel('Launched Month')
    plt.ylabel('Count')
    plt.savefig(get_path(test) + 'launched_count_over_month.jpg')
    plt.clf()


def plot_month_success_rate(data, test=False):
    '''
    Takes in a Pandas DataFrame containing the Kickstarter data and plot the
    percentage of successful project launched with respect to each month.
    '''
    data.loc[:, 'launched_month'] = \
        data.loc[:, 'launched_month'].apply(lambda x: int(x))
    data_success = data[data['state'] == 'successful']
    data_success_count = data_success.groupby('launched_month')['ID'].count()
    data_count = data.groupby('launched_month')['ID'].count()
    success_rate = data_success_count / data_count * 100
    success_rate.plot.bar()
    plt.title('Percentage of Successful Projects Launched Each Month')
    plt.xlabel('Launched Month')
    plt.ylabel('Percentage of Successful Projects (%)')
    plt.savefig(get_path(test) + 'success_rate_over_month.jpg')
    plt.clf()


def run(data):
    '''
    Takes in preprocessed Kickstarter data and performs launch time analysis.
    '''
    plot_success_fail_vs_total(data)
    plot_success_rate(data)
    plot_month_counts(data)
    plot_month_success_rate(data)


if __name__ == '__main__':
    run()

'''
This program runs functions for the Kickstarter data analysis
'''
import cse163_utils  # noqa: F401
import pandas as pd
import ks_ml
import ks_launch_time
import ks_statistical_analysis


def preprocess_data(filename):
    '''
    Takes a String filename for the Kickstarter data.
    Takes in a Pandas DataFrame containing the data from the Kickstarter
    platform. Filters down to only include projects that have launched data
    from 2010-01-01 to 2017-12-31. Calculates the pledged ratio for each
    project, extracts the launched month from the launched date index,
    calculates the duration of each campaign in days, and stores them in new
    columns.
    Returns a new DataFrame with the extra information.
    '''
    data = pd.read_csv(filename, index_col='launched', parse_dates=True)
    is_in_range = (data.index > '2010-01-01') & (data.index < '2017-12-31')
    data = data[is_in_range].copy()
    data.loc[:, 'pledged_ratio'] = data['usd_pledged_real'] \
        / data['usd_goal_real']
    data.loc[:, 'launched'] = data.index
    data.loc[:, 'launched_month'] = data.loc[:, 'launched'].apply(get_month)
    data.loc[:, 'launched'] = pd.to_datetime(data['launched'])
    data.loc[:, 'deadline'] = pd.to_datetime(data['deadline'])
    data.loc[:, 'duration'] = (data['deadline'] - data['launched']) \
        .apply(get_duration_in_days)
    return data


def get_month(date):
    '''
    Takes in a DatetimeIndex and returns the month as a String.
    '''
    return (str)(date.month)


def get_duration_in_days(timedelta):
    '''
    Takes in a TimeDelta and returns the number of days in the TimeDelta
    object as a numpy.int64.
    '''
    return timedelta.days


def main():
    '''
    Runs functions to produce data analysis results for the Kickstarter
    dataset
    '''
    data = preprocess_data('ks-projects-201801.csv')
    ks_ml.run(data)
    ks_launch_time.run(data)
    ks_statistical_analysis.run(data)


if __name__ == '__main__':
    main()

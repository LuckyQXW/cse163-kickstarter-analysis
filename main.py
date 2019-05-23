"""
This program runs functions for the KickStarter data analysis
"""
import cse163_utils  # noqa: F401
import pandas as pd


def preprocess_data(filename):
    """
    Takes a String filename for the KickStarter data.
    Reads the KickStarter data and uses the launched date for index,
    returns a pandas DataFrame that only contains projects that have
    launched date from 2010-01-01 to 2017-12-31.
    """
    data = pd.read_csv(filename, index_col='launched', parse_dates=True)
    is_in_range = (data.index > '2010-01-01') & (data.index < '2017-12-31')
    return data[is_in_range]


def main():
    """
    Runs functions to produce data analysis results for the KickStarter
    dataset
    """
    data = preprocess_data('ks-projects-201801.csv')
    data = data.sort_index()
    print(data)


if __name__ == '__main__':
    main()

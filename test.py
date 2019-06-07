'''
This file contains functions used to test the functions written for
Kickstarter data analysis
'''
from cse163_utils import assert_equals  # noqa: F401
from main import preprocess_data
import ks_launch_time


def test_plot_success_fail_vs_total(data):
    '''
    Takes in the preprocessed test dataset and tests
    plot_plot_success_fail_vs_total using the test dataset. Produces a test
    output in the test folder.
    '''
    print('Testing plot_success_fail_vs_total')
    ks_launch_time.plot_success_fail_vs_total(data, test=True)


def test_plot_success_rate(data):
    '''
    Takes in the preprocessed test dataset and tests
    plot_success_rate using the test dataset. Produces a test output
    in the test folder.
    '''
    print('Testing plot_success_rate')
    ks_launch_time.plot_success_rate(data, test=True)


def test_plot_month_counts(data):
    '''
    Takes in the preprocessed test dataset and tests
    plot_month_counts using the test dataset. Produces a test output
    in the test folder.
    '''
    print('Testing plot_month_counts')
    ks_launch_time.plot_month_counts(data, test=True)


def test_plot_month_success_rate(data):
    '''
    Takes in the preprocessed test dataset and tests
    plot_month_success_rate using the test dataset. Produces a test output
    in the test folder.
    '''
    print('Testing plot_month_success_rate')
    ks_launch_time.plot_month_success_rate(data, test=True)


def main():
    data = preprocess_data('test-dataset.csv')
    test_plot_success_fail_vs_total(data)
    test_plot_success_rate(data)
    test_plot_month_counts(data)
    test_plot_month_success_rate(data)


if __name__ == '__main__':
    main()

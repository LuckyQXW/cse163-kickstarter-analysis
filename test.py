'''
This file contains functions used to test the functions written for
Kickstarter data analysis
'''
from cse163_utils import assert_equals  # noqa: F401
from main import preprocess_data
import ks_launch_time
import statistical_analysis as sa


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


def test_cut_and_arrange(data):
    '''
    Tests the cut_and_arrange function with the given test dataset.
    '''
    print('Testing cut_and_arrange')
    expected_out = data.iloc[:1, ].to_dict()
    assert_equals(expected_out, sa.cut_and_arrange(data, 0.5).to_dict())


def test_get_unique(data):
    '''
    Tests the get_unique function with the given test dataset.
    '''
    print('Testing get_unique')
    expected_out = ['Games', 'Technology']
    assert_equals(expected_out, sa.get_unique(data, 'main_category'))


def test_get_percentages(data):
    '''
    Tests the get_percentages function with the given test dataset.
    '''
    print('Testing get_percentages')
    expected_out = {'Games': 50.0, 'Technology': 50.0}
    categories = ['Games', 'Technology']
    assert_equals(expected_out, sa.get_percentages(data, categories,
                  'main_category'))


def test_success_per_country(data):
    '''
    Tests the success_per_country function with the given test dataset.
    '''
    print('Testing success_per_country')
    expected_out = 50.0
    result = sa.success_per_country(data,
                                    'successful').to_frame().iloc[0, 0]
    assert_equals(expected_out, result)


def main():
    data = preprocess_data('test-dataset.csv')
    data2 = preprocess_data('test2-dataset.csv')

    test_plot_success_fail_vs_total(data)
    test_plot_success_rate(data)
    test_plot_month_counts(data)
    test_plot_month_success_rate(data)

    test_cut_and_arrange(data2)
    test_get_unique(data2)
    test_get_percentages(data2)
    test_success_per_country(data2)

    print('All tests passed!')


if __name__ == '__main__':
    main()

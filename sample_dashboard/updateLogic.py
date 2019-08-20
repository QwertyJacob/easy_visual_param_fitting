from django.http import JsonResponse
from sample_dashboard.chartLogic import *
from sklearn.linear_model import LinearRegression


def is_left_increment(current_param_values, param_from):
    if int(param_from) < current_param_values[0]:
        return True
    return False


def is_left_decrease(current_param_values, param_from):
    if int(param_from) > current_param_values[0]:
        return True
    return False


def is_increase_series(current_param_values, new_param_list):
    if len(new_param_list) >= len(current_param_values):
        return True
    return False


def get_new_param_list(param_from, param_to):
    new_param_list = list()
    for count in range(int(param_from), int(param_to)+1):
        new_param_list.append(count)
    return new_param_list


def get_alg_names_to_remove(values_to_remove):
    alg_names_to_remove = list()
    for value_to_remove in values_to_remove:
        alg_names_to_remove.append('Polynomial reg. deg=%s' % value_to_remove)
    return alg_names_to_remove


def get_values_to_remove(current_param_values, new_param_values):
    values_to_remove = list(set(current_param_values) - set(new_param_values))
    return values_to_remove



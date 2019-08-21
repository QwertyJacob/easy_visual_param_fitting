from sample_dashboard.chartLogic import *


def is_left_update(current_param_values, param_from, increment):
    if increment:
        return int(param_from) < current_param_values[0]
    else:
        return int(param_from) > current_param_values[0]


def is_increase_series(current_param_values, new_param_list):
    if len(new_param_list) >= len(current_param_values):
        return True
    return False


def get_decimal_cifres(float_num):
    float_array = str(float_num - int(float_num)).split('.')
    if len(float_array) > 1 :
        return len(float_array[1])
    else :
        return 0


def get_new_param_list(r):
    decimal_cifres = get_decimal_cifres(r.slider_settings.step_value)

    top_value = round(float(r.slider_settings.from_value),decimal_cifres)

    new_param_list = list()

    while top_value <= round(float(r.slider_settings.to_value),decimal_cifres) :
        new_param_list.append(top_value)
        top_value = round(top_value + round(float(r.slider_settings.step_value), decimal_cifres), decimal_cifres)

    return new_param_list


def get_alg_names_to_update(r, values_to_update):
    alg_names_to_update = list()
    for value_to_update in values_to_update:
        if r.current_predictor == predictors[0]:
            alg_names_to_update.append('Polynomial reg. deg=%s' % value_to_update)
            continue
        if r.current_predictor == predictors[1]:
            if r.current_param_to_fit == params_to_fit[0]:
                alg_names_to_update.append('SvrPoly deg=%s' % value_to_update)
                continue
            if r.current_param_to_fit == params_to_fit[1]:
                alg_names_to_update.append('SvrPoly C=%s' % value_to_update)
                continue
            if r.current_param_to_fit == params_to_fit[2]:
                alg_names_to_update.append('SvrPoly gamma=%s' % value_to_update)
                continue
            if r.current_param_to_fit == params_to_fit[3]:
                alg_names_to_update.append('SvrPoly eps=%s' % value_to_update)
                continue
        if r.current_predictor == predictors[1]:
            if r.current_param_to_fit == params_to_fit[1]:
                alg_names_to_update.append('SvrRbf C=%s' % value_to_update)
                continue
            if r.current_param_to_fit == params_to_fit[2]:
                alg_names_to_update.append('SvrRbf gamma=%s' % value_to_update)
                continue
            if r.current_param_to_fit == params_to_fit[3]:
                alg_names_to_update.append('SvrRbf eps=%s' % value_to_update)
                continue


    return alg_names_to_update


def get_values_to_update(current_param_values, new_param_values, increment):
    if increment :
        return list(set(new_param_values) - set(current_param_values))
    else:
        return list(set(current_param_values) - set(new_param_values))



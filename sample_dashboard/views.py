from django.shortcuts import render
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from django.http import JsonResponse

from sample_dashboard.chartLogic import *


def dashboard(request):
    return render(request, 'dashboard/home.html', {'context': 'empty'})


def index(request):
    return render(request, 'dashboard/welcome.html')

current_deg_values = [4, 5, 6, 7]
mean_df_week = read_data('Week15minMeanNumericalLog')
r = TestResults()


def poly_reg(request):
    global current_deg_values
    global r
    # Notice that polynomic regression is made using a transformation of the predictor values as a predictor sample for
    # linear regression.
    current_deg_values = [4, 5, 6, 7]
    r = TestResults()
    for count in range(len(current_deg_values)):
        # apply the polinomial transformation for a given degree and feature set.
        t = getLinRegTestData(mean_df_week, current_deg_values[count])

        # apply the linear regression in the transformed featurespace.
        lm = LinearRegression()
        lm.fit(t.X_train, t.y_train)
        predictions = predictLinearReg(r,t,lm)
        alg_name = 'Polynomial reg. deg=%s' % current_deg_values[count]
        processLinearReg(t, r, predictions, alg_name)

    context = getLinRegReportDump(r,mean_df_week)
    return render(request, 'dashboard/poly_reg_report.html', {'context': context})


def poly_svr_deg(request):

    r = TestResults()
    t = getSVMTestData(mean_df_week)
    deg_values = [ 4, 5, 6, 7, 8]

    for count in range(len(deg_values)):
        # Predictions are done with scaled values.
        clf = SVR(gamma=5.2, C=10, epsilon=0.1, kernel='poly', degree=deg_values[count], coef0=1)
        clf.fit(t.X_train, t.y_train)
        predictions = SVRpredict(r, t, clf)
        alg_name = 'svr poly deg=%s' % clf.degree
        processSVRReg(t, r, clf, predictions, alg_name)

    context = getSVMReportDump(t,r)

    return render(request, 'dashboard/svr_report.html', {'context': context})


def poly_svr_gamma(request):

    r = TestResults()

    t = getSVMTestData(mean_df_week)

    gamma_values = [ 5.25, 5.2, 5.1, 5]

    for count in range(len(gamma_values)):
        # Predictions are done with scaled values.
        clf = SVR(gamma=gamma_values[count], C=10, epsilon=0.1, kernel='poly', degree=7, coef0=1)
        clf.fit(t.X_train, t.y_train)
        predictions = SVRpredict(r, t, clf)
        alg_name = 'SVR poly gamma=%s' % clf.gamma
        processSVRReg(t, r, clf, predictions, alg_name)

    context = getSVMReportDump(t, r)

    return render(request, 'dashboard/svr_report.html', {'context': context})


def poly_svr_epsilon(request):

    r = TestResults()
    t = getSVMTestData(mean_df_week)
    epsilon_values = [0.1, 0.095, 0.09, 0.085]

    for count in range(len(epsilon_values)):
        # Predictions are done with scaled values.
        clf = SVR(gamma=5.2, C=10, epsilon=epsilon_values[count], kernel='poly', degree=7, coef0=1)
        clf.fit(t.X_train, t.y_train)
        predictions = SVRpredict(r, t, clf)
        alg_name = 'SVR poly7 epsilon=%s' % clf.epsilon
        processSVRReg(t, r, clf, predictions, alg_name)

    context = getSVMReportDump(t, r)

    return render(request, 'dashboard/svr_report.html', {'context': context})


def poly_svr_C(request):

    r = TestResults()
    t = getSVMTestData(mean_df_week)
    C_values = [ 1, 5, 10, 15, 20]

    for count in range(len(C_values)):
        # Predictions are done with scaled values.
        clf = SVR(gamma=5.2, C=C_values[count], epsilon=0.1, kernel='poly', degree=7, coef0=1)
        clf.fit(t.X_train, t.y_train)
        predictions = SVRpredict(r, t, clf)
        alg_name = 'SVR poly C=%s' % clf.C
        processSVRReg(t, r, clf, predictions, alg_name)

    context = getSVMReportDump(t, r)

    return render(request, 'dashboard/svr_report.html', {'context': context})


def rbf_svr_gamma(request):
    r = TestResults()
    t = getSVMTestData(mean_df_week)
    gamma_values = [5.25, 5.2, 5.1, 5]

    for count in range(len(gamma_values)):
        # Predictions are done with scaled values.
        clf = SVR(gamma=gamma_values[count], C=10, epsilon=0.1, kernel='poly', degree=7, coef0=1)
        clf.fit(t.X_train, t.y_train)
        predictions = SVRpredict(r, t, clf)
        alg_name = 'SVR RBF gamma=%s' % clf.gamma
        processSVRReg(t, r, clf, predictions, alg_name)

    context = getSVMReportDump(t, r)

    return render(request, 'dashboard/svr_report.html', {'context': context})


def rbf_svr_epsilon(request):
    r = TestResults()
    t = getSVMTestData(mean_df_week)
    epsilon_values = [0.01,0.03,0.05,0.08,0.1]

    for count in range(len(epsilon_values)):
        # Predictions are done with scaled values.
        clf = SVR(gamma=20, C=0.9, epsilon=epsilon_values[count])
        clf.fit(t.X_train, t.y_train)
        predictions = SVRpredict(r, t, clf)
        alg_name = ' SVR (RBF C=0.9) | epsilon value =%s ' % clf.epsilon
        processSVRReg(t, r, clf, predictions, alg_name)

    context = getSVMReportDump(t, r)

    return render(request, 'dashboard/svr_report.html', {'context': context})


def rbf_svr_C(request):
    r = TestResults()
    t = getSVMTestData(mean_df_week)
    C_values = [0.1,0.6,1,1.5]

    for count in range(len(C_values)):
        # Predictions are done with scaled values.
        clf = SVR(gamma=20, C=C_values[count])
        clf.fit(t.X_train, t.y_train)
        predictions = SVRpredict(r, t, clf)
        alg_name = 'SVR Rbf-k,  gamma=20, C value=%s' % clf.C
        processSVRReg(t, r, clf, predictions, alg_name)

    context = getSVMReportDump(t, r)

    return render(request, 'dashboard/svr_report.html', {'context': context})


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


def update_report(request):
    global current_deg_values
    global r

    degree_to = request.GET.get('to', None)
    degree_from = request.GET.get('from', None)

    new_param_list = get_new_param_list(degree_from, degree_to)

    increase_series = is_increase_series(current_deg_values, new_param_list)

    if increase_series:
        local_left_increment = is_left_increment(current_deg_values, degree_from)
        return increase_info_in_linear_report(new_param_list, current_deg_values, local_left_increment)
    else:
        return remove_info_from_linear_report(current_deg_values, new_param_list)


def remove_info_from_linear_report(current_param_values, new_param_values):
    global current_deg_values
    global r

    r.left_decrease = is_left_decrease(current_param_values, new_param_values[0])
    values_to_remove = get_values_to_remove(current_param_values, new_param_values)
    alg_names_to_remove = get_alg_names_to_remove(values_to_remove)
    updateTestResults_decrease(values_to_remove, alg_names_to_remove)
    context = get_update_report_dump_decrease(r, alg_names_to_remove)

    current_deg_values = new_param_values
    return JsonResponse(context)


def get_alg_names_to_remove(values_to_remove):
    alg_names_to_remove = list()
    for value_to_remove in values_to_remove:
        alg_names_to_remove.append('Polynomial reg. deg=%s' % value_to_remove)
    return alg_names_to_remove


def updateTestResults_decrease(values_to_remove, alg_names_to_remove):
    global r

    for value_to_remove in values_to_remove:
        if r.left_decrease:
            r.norm_metrics_df.drop(r.norm_metrics_df.head(1).index,inplace=True)
            r.raw_metrics_df.drop(r.raw_metrics_df.head(1).index,inplace=True)
            r.mean_absolute_errors.pop(0)
            r.mean_squared_errors.pop(0)
            r.times_list.pop(0)
            r.std_dev_list.pop(0)
            r.report_df_list.pop(0)
            r.error_ds.pop(0)

        else:
            r.norm_metrics_df.drop(r.norm_metrics_df.tail(1).index,inplace=True)
            r.raw_metrics_df.drop(r.raw_metrics_df.tail(1).index,inplace=True)
            r.mean_absolute_errors = r.mean_absolute_errors[:-1]
            r.mean_squared_errors = r.mean_squared_errors[:-1]
            r.times_list = r.times_list[:-1]
            r.std_dev_list = r.std_dev_list[:-1]
            r.report_df_list = r.report_df_list[:-1]
            r.error_ds = r.error_ds[:-1]

    r.alg_names_list = list(set(r.alg_names_list) - set(alg_names_to_remove))
    r.alg_names_list.sort(key=sortDegree)


def get_values_to_remove(current_param_values, new_param_values):
    values_to_remove = list(set(current_param_values) - set(new_param_values))
    return values_to_remove


def increase_info_in_linear_report(new_param_list, current_param_values,local_left_increment):
    global current_deg_values
    global r

    new_alg_names = list()
    new_values = list(set(new_param_list) - set(current_param_values))
    if local_left_increment:
        new_values.reverse()

    current_deg_values = new_param_list

    for new_value in new_values:
        # apply the polynomial transformation for a given degree and feature set.
        t = getLinRegTestData(mean_df_week, new_value)
        t.left_increment = local_left_increment
        # apply the linear regression in the transformed feature space.
        lm = LinearRegression()
        lm.fit(t.X_train, t.y_train)
        predictions = predictLinearReg(r, t, lm)
        alg_name = 'Polynomial reg. deg=%s' % new_value
        new_alg_names.append(alg_name)
        processLinearReg(t, r, predictions, alg_name)

    context = getLinRegReportDumpDelta(r, new_alg_names)

    return JsonResponse(context)
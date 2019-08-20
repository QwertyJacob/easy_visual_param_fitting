from django.shortcuts import render
from sklearn.svm import SVR
from sample_dashboard.updateLogic import *

########### GLOBAL VARS ###############################
current_deg_values = [4, 5, 6, 7]
mean_df_week = read_data('Week15minMeanNumericalLog')
r = TestResults()

#######################################################

def dashboard(request):
    return render(request, 'dashboard/home.html', {'context': 'empty'})


def index(request):
    return render(request, 'dashboard/welcome.html')


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
        predictions = predictLinearReg(r, t, lm)
        alg_name = 'Polynomial reg. deg=%s' % current_deg_values[count]
        processLinearReg(t, r, predictions, alg_name)

    context = getLinRegReportDump(r, mean_df_week)
    return render(request, 'dashboard/poly_reg_report.html', {'context': context})


def poly_svr_deg(request):
    r = TestResults()
    t = getSVMTestData(mean_df_week)
    deg_values = [4, 5, 6, 7, 8]

    for count in range(len(deg_values)):
        # Predictions are done with scaled values.
        clf = SVR(gamma=5.2, C=10, epsilon=0.1, kernel='poly', degree=deg_values[count], coef0=1)
        clf.fit(t.X_train, t.y_train)
        predictions = SVRpredict(r, t, clf)
        alg_name = 'svr poly deg=%s' % clf.degree
        processSVRReg(t, r, clf, predictions, alg_name)

    context = getSVMReportDump(t, r)

    return render(request, 'dashboard/svr_report.html', {'context': context})


def poly_svr_gamma(request):
    r = TestResults()

    t = getSVMTestData(mean_df_week)

    gamma_values = [5.25, 5.2, 5.1, 5]

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
    C_values = [1, 5, 10, 15, 20]

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
    epsilon_values = [0.01, 0.03, 0.05, 0.08, 0.1]

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
    C_values = [0.1, 0.6, 1, 1.5]

    for count in range(len(C_values)):
        # Predictions are done with scaled values.
        clf = SVR(gamma=20, C=C_values[count])
        clf.fit(t.X_train, t.y_train)
        predictions = SVRpredict(r, t, clf)
        alg_name = 'SVR Rbf-k,  gamma=20, C value=%s' % clf.C
        processSVRReg(t, r, clf, predictions, alg_name)

    context = getSVMReportDump(t, r)

    return render(request, 'dashboard/svr_report.html', {'context': context})


def update_report(request):
    global current_deg_values
    global r

    degree_to = request.GET.get('to', None)
    degree_from = request.GET.get('from', None)

    new_param_list = get_new_param_list(degree_from, degree_to)

    increase_series = is_increase_series(current_deg_values, new_param_list)

    if increase_series:
        return increase_info_in_linear_report(new_param_list, current_deg_values)
    else:
        return remove_info_from_linear_report(current_deg_values, new_param_list)



def remove_info_from_linear_report(current_param_values, new_param_values):
    global current_deg_values
    global r

    r.left_update = is_left_decrease(current_param_values, new_param_values[0])
    values_to_remove = get_values_to_remove(current_param_values, new_param_values)
    alg_names_to_remove = get_alg_names_to_remove(values_to_remove)
    updateTestResults(values_to_remove, alg_names_to_remove, False)
    context = get_update_report_dump(r, alg_names_to_remove, False)

    current_deg_values = new_param_values
    return JsonResponse(context)


def increase_info_in_linear_report(new_param_list, current_param_values):
    global current_deg_values
    global r
    new_alg_names = list()

    r.left_update = is_left_increment(current_deg_values, new_param_list[0])
    new_values = list(set(new_param_list) - set(current_param_values))
    if r.left_update:
        new_values.reverse()
    updateTestResults(new_values, new_alg_names, True)
    context = get_update_report_dump(r, new_alg_names, True)

    current_deg_values = new_param_list
    return JsonResponse(context)


def updateTestResults(new_values, new_alg_names, increase):
    if increase:
        update_test_results_increase(new_values, new_alg_names)
    else :
        update_test_results_decrease(new_values, new_alg_names)



def update_test_results_increase(new_values, new_alg_names):
    global r
    global mean_df_week

    for new_value in new_values:
        # apply the polynomial transformation for a given degree and feature set.
        t = getLinRegTestData(mean_df_week, new_value)
        # apply the linear regression in the transformed feature space.
        lm = LinearRegression()
        lm.fit(t.X_train, t.y_train)
        predictions = predictLinearReg(r, t, lm)
        alg_name = 'Polynomial reg. deg=%s' % new_value
        new_alg_names.append(alg_name)
        processLinearReg(t, r, predictions, alg_name)


def update_test_results_decrease(new_values, new_alg_names):
    global r
    for value_to_remove in new_values:
        if r.left_update:
            r.norm_metrics_df.drop(r.norm_metrics_df.head(1).index, inplace=True)
            r.raw_metrics_df.drop(r.raw_metrics_df.head(1).index, inplace=True)
            r.mean_absolute_errors.pop(0)
            r.mean_squared_errors.pop(0)
            r.times_list.pop(0)
            r.std_dev_list.pop(0)
            r.report_df_list.pop(0)
            r.error_ds.pop(0)

        else:
            r.norm_metrics_df.drop(r.norm_metrics_df.tail(1).index, inplace=True)
            r.raw_metrics_df.drop(r.raw_metrics_df.tail(1).index, inplace=True)
            r.mean_absolute_errors = r.mean_absolute_errors[:-1]
            r.mean_squared_errors = r.mean_squared_errors[:-1]
            r.times_list = r.times_list[:-1]
            r.std_dev_list = r.std_dev_list[:-1]
            r.report_df_list = r.report_df_list[:-1]
            r.error_ds = r.error_ds[:-1]

    r.alg_names_list = list(set(r.alg_names_list) - set(new_alg_names))
    r.alg_names_list.sort(key=sortDegree)

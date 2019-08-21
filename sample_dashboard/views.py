from django.shortcuts import render
from sklearn.svm import SVR
from sample_dashboard.updateLogic import *
from sklearn.linear_model import LinearRegression
from django.http import JsonResponse

########### GLOBAL VARS ###############################
mean_df_week = read_data('Week15minMeanNumericalLog')
r = TestResults()
t = getSVMTestData(mean_df_week)

initPolyRegSS = SliderSettings(1, 20, 1, 4, 7)
polysvrdegSS = SliderSettings(1, 20, 1, 4, 7)
polysvrgammaSS = SliderSettings(4.0, 6.0, 0.1, 5.0, 5.2) #hot values: [5.25, 5.2, 5.1, 5]
polysvrepsilonSS = SliderSettings(0.05, 0.15, 0.005, 0.085, 0.1)
polysvrCSS = SliderSettings(0, 100, 5, 1, 20)
rbfsvrgammaSS = SliderSettings(5.1, 5.3, 0.01, 5.18, 5.21)
rbfsvrepsilonSS = SliderSettings(0.01, 0.2, 0.03, 0.03, 0.12) #hot values [0.01, 0.03, 0.05, 0.08, 0.1]
rbfsvrCSS = SliderSettings(0.1, 10, 0.5, 0.1, 1.5)


#######################################################

def dashboard(request):
    return render(request, 'dashboard/home.html', {'context': 'empty'})


def index(request):
    return render(request, 'dashboard/welcome.html')


def poly_reg(request):
    global r
    # Notice that polynomic regression is made using a transformation of the predictor values as a predictor sample for
    # linear regression.
    r = TestResults()
    r.slider_settings = initPolyRegSS
    r.current_param_values = get_new_param_list(r)
    r.current_predictor = predictors[0]
    r.current_param_to_fit = params_to_fit[0]

    for count in range(len(r.current_param_values)):
        # apply the polinomial transformation for a given degree and feature set.
        t_linear = getLinRegTestData(mean_df_week, int(r.current_param_values[count]))

        # apply the linear regression in the transformed featurespace.
        lm = LinearRegression()
        lm.fit(t_linear.X_train, t_linear.y_train)
        predictions = predictLinearReg(r, t_linear, lm)
        alg_name = 'Polynomial reg. deg=%s' % r.current_param_values[count]
        processLinearReg(t_linear, r, predictions, alg_name)

    context = getLinRegReportDump(r, mean_df_week)
    return render(request, 'dashboard/poly_reg_report.html', {'context': context})


def poly_svr_deg(request):
    global r
    r = TestResults()
    r.slider_settings = polysvrdegSS
    r.current_param_values = get_new_param_list(r)
    r.current_predictor = predictors[1]
    r.current_param_to_fit = params_to_fit[0]

    for count in range(len(r.current_param_values)):
        # Predictions are done with scaled values.
        clf = SVR(gamma=5.2, C=10, epsilon=0.1, kernel='poly', degree=r.current_param_values[count], coef0=1)
        clf.fit(t.X_train, t.y_train)
        predictions = SVRpredict(r, t, clf)
        alg_name = 'SvrPoly deg=%s' % clf.degree
        processSVRReg(t, r, clf, predictions, alg_name)

    context = getSVMReportDump(t, r)

    return render(request, 'dashboard/svr_report.html', {'context': context})


def poly_svr_gamma(request):
    global r
    r = TestResults()
    r.slider_settings = polysvrgammaSS
    r.current_param_values = get_new_param_list(r)
    r.current_predictor = predictors[1]
    r.current_param_to_fit = params_to_fit[2]

    for count in range(len(r.current_param_values)):
        # Predictions are done with scaled values.
        clf = SVR(gamma=r.current_param_values[count], C=10, epsilon=0.1, kernel='poly', degree=7, coef0=1)
        clf.fit(t.X_train, t.y_train)
        predictions = SVRpredict(r, t, clf)
        alg_name = 'SvrPoly gamma=%s' % clf.gamma
        processSVRReg(t, r, clf, predictions, alg_name)

    context = getSVMReportDump(t, r)

    return render(request, 'dashboard/svr_report.html', {'context': context})


def poly_svr_epsilon(request):
    global r
    r = TestResults()
    r.slider_settings = polysvrepsilonSS
    r.current_param_values = get_new_param_list(r)
    r.current_predictor = predictors[1]
    r.current_param_to_fit = params_to_fit[3]

    for count in range(len(r.current_param_values)):
        # Predictions are done with scaled values.
        clf = SVR(gamma=5.2, C=10, epsilon=r.current_param_values[count], kernel='poly', degree=7, coef0=1)
        clf.fit(t.X_train, t.y_train)
        predictions = SVRpredict(r, t, clf)
        alg_name = 'SvrPoly eps=%s' % clf.epsilon
        processSVRReg(t, r, clf, predictions, alg_name)

    context = getSVMReportDump(t, r)

    return render(request, 'dashboard/svr_report.html', {'context': context})


def poly_svr_C(request):
    global r
    r = TestResults()
    r.slider_settings = polysvrCSS
    r.current_param_values = get_new_param_list(r)
    r.current_predictor = predictors[1]
    r.current_param_to_fit = params_to_fit[1]

    for count in range(len(r.current_param_values)):
        # Predictions are done with scaled values.
        clf = SVR(gamma=5.2, C=r.current_param_values[count], epsilon=0.1, kernel='poly', degree=7, coef0=1)
        clf.fit(t.X_train, t.y_train)
        predictions = SVRpredict(r, t, clf)
        alg_name = 'SvrPoly C=%s' % clf.C
        processSVRReg(t, r, clf, predictions, alg_name)

    context = getSVMReportDump(t, r)

    return render(request, 'dashboard/svr_report.html', {'context': context})


def rbf_svr_gamma(request):
    global r
    r = TestResults()
    r.slider_settings = rbfsvrgammaSS
    r.current_param_values = get_new_param_list(r)
    r.current_predictor = predictors[2]
    r.current_param_to_fit = params_to_fit[2]

    for count in range(len(r.current_param_values)):
        # Predictions are done with scaled values.
        clf = SVR(gamma=r.current_param_values[count], C=10, epsilon=0.1, kernel='poly', degree=7, coef0=1)
        clf.fit(t.X_train, t.y_train)
        predictions = SVRpredict(r, t, clf)
        alg_name = 'SvrRbf gamma=%s' % clf.gamma
        processSVRReg(t, r, clf, predictions, alg_name)

    context = getSVMReportDump(t, r)

    return render(request, 'dashboard/svr_report.html', {'context': context})


def rbf_svr_epsilon(request):
    global r
    r = TestResults()
    r.slider_settings = rbfsvrepsilonSS
    r.current_param_values = get_new_param_list(r)
    r.current_predictor = predictors[2]
    r.current_param_to_fit = params_to_fit[3]

    for count in range(len(r.current_param_values)):
        # Predictions are done with scaled values.
        clf = SVR(gamma=20, C=0.9, epsilon=r.current_param_values[count])
        clf.fit(t.X_train, t.y_train)
        predictions = SVRpredict(r, t, clf)
        alg_name = 'SvrRbf eps=%s' % clf.epsilon
        processSVRReg(t, r, clf, predictions, alg_name)

    context = getSVMReportDump(t, r)

    return render(request, 'dashboard/svr_report.html', {'context': context})


def rbf_svr_C(request):
    global r
    r = TestResults()
    r.slider_settings = rbfsvrCSS
    r.current_param_values = get_new_param_list(r)
    r.current_predictor = predictors[2]
    r.current_param_to_fit = params_to_fit[1]

    for count in range(len(r.current_param_values)):
        # Predictions are done with scaled values.
        clf = SVR(gamma=20, C=r.current_param_values[count])
        clf.fit(t.X_train, t.y_train)
        predictions = SVRpredict(r, t, clf)
        alg_name = 'SvrRbf C=%s' % clf.C
        processSVRReg(t, r, clf, predictions, alg_name)

    context = getSVMReportDump(t, r)

    return render(request, 'dashboard/svr_report.html', {'context': context})


def update_report(request):
    global r

    param_to = request.GET.get('to', None)
    param_from = request.GET.get('from', None)
    r.current_predictor = request.GET.get('predictor', None)
    r.current_param_to_fit = request.GET.get('param_to_fit', None)
    r.slider_settings.to_value = param_to
    r.slider_settings.from_value = param_from

    new_param_list = get_new_param_list(r)

    increase_series = is_increase_series(r.current_param_values, new_param_list)

    return update_report_logic(new_param_list, increase_series)


def update_report_logic(new_param_values, increase):
    global r

    r.left_update = is_left_update(r.current_param_values, new_param_values[0], increase)
    values_to_update = get_values_to_update(r.current_param_values, new_param_values, increase)
    alg_names_to_update = get_alg_names_to_update(r,values_to_update)
    updateTestResults(r, values_to_update, alg_names_to_update, increase)
    context = get_update_report_dump(r, alg_names_to_update, increase)
    r.current_param_values = new_param_values

    return JsonResponse(context)


def updateTestResults(r, new_values, new_alg_names, increase):
    if increase:
        if r.left_update:
            new_values.reverse()
            new_alg_names.reverse()
        if r.current_predictor == predictors[0]:
            update_linear_test_results_increase(new_values, new_alg_names)
        if r.current_predictor == predictors[1]:
            update_svr_test_results_increase(new_values, new_alg_names)
    else:
        update_test_results_decrease(new_values, new_alg_names)


def update_svr_test_results_increase(new_values, new_alg_names):
    global r
    global t
    index = 0
    for new_value in new_values:
        # Predictions are done with scaled values.
        clf = SVR(gamma=5.2, C=10, epsilon=0.1,
                  kernel='poly', degree=new_value, coef0=1)
        clf.fit(t.X_train, t.y_train)
        predictions = SVRpredict(r, t, clf)
        processSVRReg(t, r, clf, predictions, new_alg_names[index])
        index = index + 1


def update_linear_test_results_increase(new_values, new_alg_names):
    global r
    global mean_df_week
    index = 0
    for new_value in new_values:
        # apply the polynomial transformation for a given degree and feature set.
        t_linear = getLinRegTestData(mean_df_week, new_value)
        # apply the linear regression in the transformed feature space.
        lm = LinearRegression()
        lm.fit(t_linear.X_train, t_linear.y_train)
        predictions = predictLinearReg(r, t_linear, lm)
        processLinearReg(t_linear, r, predictions, new_alg_names[index])
        index = index + 1


def update_test_results_decrease(new_values, new_alg_names):
    global r

    for value_to_remove in new_values:
        if r.left_update:
            r.norm_metrics_df.drop(r.norm_metrics_df.head(1).index, inplace=True)
            r.raw_metrics_df.drop(r.raw_metrics_df.head(1).index, inplace=True)
            r.predictions_df_list.pop(0)
            r.mean_absolute_errors.pop(0)
            r.mean_squared_errors.pop(0)
            r.times_list.pop(0)
            r.std_dev_list.pop(0)
            r.report_df_list.pop(0)
            r.error_ds.pop(0)
            if r.current_predictor != predictors[0]:
               r.sv_ratio_list.pop(0)
        else:
            r.norm_metrics_df.drop(r.norm_metrics_df.tail(1).index, inplace=True)
            r.raw_metrics_df.drop(r.raw_metrics_df.tail(1).index, inplace=True)
            r.mean_absolute_errors = r.mean_absolute_errors[:-1]
            r.mean_squared_errors = r.mean_squared_errors[:-1]
            r.times_list = r.times_list[:-1]
            r.std_dev_list = r.std_dev_list[:-1]
            r.report_df_list = r.report_df_list[:-1]
            r.error_ds = r.error_ds[:-1]
            r.predictions_df_list = r.predictions_df_list[:-1]
            if r.current_predictor != predictors[0]:
               r.sv_ratio_list =  r.sv_ratio_list[:-1]

    r.alg_names_list = list(set(r.alg_names_list) - set(new_alg_names))
    r.alg_names_list.sort(key=sortDegree)

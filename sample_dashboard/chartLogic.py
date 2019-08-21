import numpy as np
import pandas as pd
import json
import timeit
import statistics
import re

from dominate.tags import div, li, ul, a, table, thead, tr, th, td, tbody, label, span, input
from sklearn.preprocessing import PolynomialFeatures
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
from datetime import datetime


class TestDictionary:
    def __init__(self, paramToFit, values):
        self.paramToFit = paramToFit
        self.values = values

class TestResults:
    def __init__(self):
        self.predictions_df_list = list()
        self.mean_absolute_errors = list()
        self.mean_squared_errors = list()
        self.sv_ratio_list = list()
        self.std_dev_list = list()
        self.times_list = list()
        self.error_ds = list()
        self.alg_names_list = list()
        self.report_df_list = list()
        self.norm_metrics_df = []
        self.raw_metrics_df = []
        self.left_update = False
        self.current_predictor = ''
        self.current_param_to_fit = ''
        self.current_param_values = list()
        self.slider_settings = None

class SVMTestData:
    def __init__(self, X_train, X_test, y_train, y_test, original_df, svmScaler, svmDf_scaled):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.original_df = original_df
        self.svmScaler = svmScaler
        self.svmDf_scaled = svmDf_scaled


class LinRegTestData:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test


class SliderSettings:
    def __init__(self, min_value, max_value, step_value, from_value, to_value):
        self.from_value = from_value
        self.to_value = to_value
        self.step_value = step_value
        self.min_value = min_value
        self.max_value = max_value




linear_metric_names = ['mean_abs_err', 'mean_sqrd_err', 'std_dev', 'performance (negative)']
svr_metric_names = ['mean_abs_err', 'mean_sqrd_err', 'std_dev', 'performance (negative)', 'sv ratio']
predictors = ['Polynomial Regression', 'SVR with polynomial Kernel', 'SVR with RBF kernel']
params_to_fit = ['deg','C','gamma','epsilon']

def getSVMReportDump(t,r):

    ground_thruth_data = t.svmDf_scaled.tolist()

    predictionsChart = get_predictions_chart(ground_thruth_data, r.alg_names_list, r.predictions_df_list)

    #errorDistrosChart = getErrorDistrosChart(r.predictions_df_list, r.error_ds, r.alg_names_list)

    r.raw_metrics_df = get_raw_metrics_df(r, svr_metric_names)
    # we should save somewhere  absolute values for a final exaustive comparison:
    r.norm_metrics_df = get_norm_metrics_df(r, svr_metric_names)

    norm_metricsChart = getNormalizedMetricsChart(svr_metric_names, r)

    rawMetricsChart = getRawMetricsChart(svr_metric_names, r.raw_metrics_df, r.alg_names_list)

    table_content = getTableTabsDiv(r)

    error_charts = get_error_charts(r)

    tab_names = get_tab_names(r)

    #alg_filter_buttons_div = get_alg_filter_buttons_div(r, tab_names)

    context = {'predictionsChart': predictionsChart,
               'metricsChart': norm_metricsChart,
               'rawMetricsChart': rawMetricsChart,
               'table_data': table_content,
               'error_charts': error_charts,
               'alg_names': tab_names,
               'predictor': r.current_predictor,
               'param_to_fit': r.current_param_to_fit,
               'current_param_values' : r.current_param_values }

    return context


def get_alg_filter_buttons_div(r, tab_names):
    alg_filter_buttons_div = div(cls="container")

    alg_input_tag = input(id="ground_truth_CB", cls="badgebox")
    alg_input_tag.attributes['type'] = "checkbox"
    alg_span_tag = span("✓", cls="badge", id="ground_truth_CB_Span")

    alg_filter_button = label("Ground Truth", cls="btn", id="ground_truth_CB_Label")
    alg_filter_button.appendChild(alg_input_tag)
    alg_filter_button.appendChild(alg_span_tag)
    alg_filter_buttons_div.appendChild(alg_filter_button)

    for count in range(len(tab_names)):
        alg_input_tag = input(id=tab_names[count]+"_CB", cls="badgebox")
        alg_input_tag.attributes['type']="checkbox"
        alg_span_tag = span("✓",cls="badge", id=tab_names[count]+"_CB_Span")

        alg_filter_button = label(tab_names[count], cls="btn", id=tab_names[count]+"_CB_Label")
        alg_filter_button.appendChild(alg_input_tag)
        alg_filter_button.appendChild(alg_span_tag)

        alg_filter_buttons_div.appendChild(alg_filter_button)
    return alg_filter_buttons_div.render()


def get_error_charts(r):
    error_charts = list()
    for count in range(len(r.alg_names_list)):
        error_charts.append(get_single_errorDistroChart(r.error_ds[count].iloc[:,0].tolist(), r.alg_names_list[count]))
    return error_charts


def get_tab_names(r):
    tab_names = list()
    for count in range(len(r.report_df_list)):
        tab_names.append(getFormattedName(r.alg_names_list[count]))
    return tab_names


def getFormattedName(str):
    return str.replace(".","").replace(" ","").replace("=","").replace("|","").replace("(","").replace(")","")


def getTableTabsDiv(r):

    table_tabs_div = div(['contentHolder','contentHolder'], cls='container')

    tablist = ul(cls='nav nav-pills mb-3')
    tablist.attributes['role']="tablist"

    tab_names = list()
    for count in range(len(r.report_df_list)):
        tab_names.append(getFormattedName(r.alg_names_list[count]))

    for count in range(len(r.report_df_list)):
        if (count == 0):
            tab = a(r.alg_names_list[count], cls="nav-link active", href='#'+tab_names[count]+"-tab-pane")
        else:
            tab = a(r.alg_names_list[count], href='#' + tab_names[count]+"-tab-pane", cls="nav-link")

        tab.attributes['data-toggle']="pill"
        li_element = li(tab,cls='nav-item')
        tablist += li_element

    table_tabs_div[0] = tablist

    tabContentDiv = div(cls='tab-content')
    for count in range(len(r.report_df_list)):
        if (count == 0):
            tabdiv = div(getDescriptionPills(r.report_df_list[count]['Error'].describe()), id=tab_names[count] + "-tab-pane", cls='tab-pane active')
        else:
            tabdiv = div(getDescriptionPills(r.report_df_list[count]['Error'].describe()), id=tab_names[count] + "-tab-pane", cls='tab-pane fade')
        tabdiv.appendChild(div('contentHolder',id=tab_names[count]+"-chart-div"))
        tabdiv.appendChild(div(getErrorTableContent(r.report_df_list[count])))
        tabContentDiv.appendChild(tabdiv)

    table_tabs_div[1] = tabContentDiv

    return table_tabs_div.render()


def getDescriptionPills(statistics):
    statistics = np.round(statistics,1)

    statisticstable = table(cls='table')
    table_header_row = tr()
    table_header_row.appendChild(th('Measurements count'))
    table_header_row.appendChild(th('Mean error'))
    table_header_row.appendChild(th('Standard Error dev'))
    table_header_row.appendChild(th('Min error measured'))
    table_header_row.appendChild(th('First quartile (25%)'))
    table_header_row.appendChild(th('Median (50%)'))
    table_header_row.appendChild(th('Third quartile (75%)'))
    table_header_row.appendChild(th('Max error measured'))

    table_row = tr()
    table_row.appendChild(td(str(statistics[0])))
    table_row.appendChild(td(str(statistics[1])))
    table_row.appendChild(td(str(statistics[2])))
    table_row.appendChild(td(str(statistics[3])))
    table_row.appendChild(td(str(statistics[4])))
    table_row.appendChild(td(str(statistics[5])))
    table_row.appendChild(td(str(statistics[6])))
    table_row.appendChild(td(str(statistics[7])))

    statisticstable.appendChild(thead(table_header_row))
    statisticstable.appendChild(tbody(table_row))
    return statisticstable


def get_raw_metrics_df(r, metric_names):
    if len(metric_names) == 4:
        metric_list = list(zip(r.mean_absolute_errors, r.mean_squared_errors, r.std_dev_list, r.times_list))
    else:
        metric_list = list(zip(r.mean_absolute_errors, r.mean_squared_errors, r.std_dev_list, r.times_list, r.sv_ratio_list))

    poly_svr_metricsDf = pd.DataFrame(metric_list,
        index=r.alg_names_list,
        columns=metric_names)
    return poly_svr_metricsDf


def get_norm_metrics_df(r, metric_names):

    scaler = preprocessing.MaxAbsScaler()
    metricsDF_scaler = scaler.fit(r.raw_metrics_df)
    scaled_metrics_arr = metricsDF_scaler.transform(r.raw_metrics_df)

    scaled_metrics_arr = np.around(scaled_metrics_arr, decimals=2)

    norm_metrics_df = r.raw_metrics_df.copy()
    for count in range(len(metric_names)):
        norm_metrics_df[metric_names[count]] = column(scaled_metrics_arr, count)
    return norm_metrics_df


def getLinRegReportDump(r, df):

    ground_thruth_data = df[['HHMM', 'bandwidth']].values.tolist()

    predictionsChart = get_predictions_chart(ground_thruth_data, r.alg_names_list, r.predictions_df_list)

    r.raw_metrics_df = get_raw_metrics_df(r, linear_metric_names)

    r.norm_metrics_df = get_norm_metrics_df(r, linear_metric_names)

    norm_metricsChart = getNormalizedMetricsChart(linear_metric_names, r)

    rawMetricsChart = getRawMetricsChart(linear_metric_names, r.raw_metrics_df, r.alg_names_list)

    table_content = getTableTabsDiv(r)

    error_charts = get_error_charts(r)

    tab_names = get_tab_names(r)

    context = {'predictionsChart': predictionsChart,
               'metricsChart': norm_metricsChart,
               'rawMetricsChart': rawMetricsChart,
               'table_data': table_content,
               'error_charts': error_charts,
               'alg_names': tab_names,
               'predictor': r.current_predictor,
               'param_to_fit': r.current_param_to_fit,
               'current_param_values' : r.current_param_values}

    return context

def sortDegree(alg_name):
    return re.findall(r'\d+', alg_name)[0]


def get_update_report_dump(r, new_alg_names, increase):

    predictionsToAppend = list()
    normMetricsToAppend = list()
    rawMetricsToAppend = list()

    if increase:
        r.raw_metrics_df = get_raw_metrics_df(r, linear_metric_names)
        r.norm_metrics_df = get_norm_metrics_df(r, linear_metric_names)

        for alg_count in range(len(new_alg_names)):
            predictionsToAppend.append({
                'type': 'line',
                'name': new_alg_names[alg_count],
                'id': new_alg_names[alg_count],
                'data': r.predictions_df_list[len(r.alg_names_list) - len(new_alg_names)].values.tolist(),
                'lineWidth': 3,
                'marker': {
                    'enabled': False
                },
                'states': {
                    'hover': {
                        'lineWidth': 4
                    }
                },
                'enableMouseTracking': True
            })

    for metric_count in range(len(linear_metric_names)):
        normMetricsToAppend.append(
            {
                'name': linear_metric_names[metric_count],
                'data': r.norm_metrics_df[linear_metric_names[metric_count]].tolist()
            }
        )
        rawMetricsToAppend.append(
            {
                'name': linear_metric_names[metric_count],
                'data': r.raw_metrics_df[linear_metric_names[metric_count]].tolist(),
                'visible': True
            }
        )
        rawMetricsToAppend.append(
            {
                'name': linear_metric_names[metric_count] + "line",
                'showInLegend': False,
                'data': r.raw_metrics_df[linear_metric_names[metric_count]].tolist(),
                'visible': True,
                'type': 'line'
            })

    tab_names = get_tab_names(r)

    context = {'increase': increase,
               'alg_names_list': r.alg_names_list,
               'predictionsToAppend': predictionsToAppend,
               'normMetricsToAppend': normMetricsToAppend,
               'rawMetricsToAppend': rawMetricsToAppend,
               'table_data': getTableTabsDiv(r),
               'errorCharts' : get_error_charts(r),
               'alg_names': tab_names}

    if not increase:
        context['alg_names_to_remove'] = new_alg_names

    return context


def SVRpredict(r, t, clf):
    # Scaled values permit us to optimize performance
    start_time = timeit.default_timer()
    predictions = clf.predict(t.X_test)
    timetook = (timeit.default_timer() - start_time)
    testResultAppend(r.times_list,timetook, r.left_update)
    return predictions

def testResultAppend(list, new_series, append_left):
    if(append_left):
        list.insert(0,new_series)
    else:
        list.append(new_series)


def predictLinearReg(r,t,lm):
    # performance measures for the prediction task
    start_time = timeit.default_timer()
    predictions = lm.predict(t.X_test)
    predictions_df = pd.DataFrame({'HHMM': t.X_test[:, 1], 'bandwidth': predictions})
    predictions_df = predictions_df.sort_values(by=['HHMM'])
    testResultAppend(r.predictions_df_list, predictions_df, r.left_update)
    timetook = (timeit.default_timer() - start_time)
    testResultAppend(r.times_list,timetook, r.left_update)
    return predictions


def processSVRReg(t, r, clf, predictions, alg_name):

    test_df = pd.concat([t.X_test, t.y_test], axis=1)
    predictions_df = test_df.copy()
    predictions_df['bandwidth'] = predictions
    predictions_df = predictions_df.sort_values(by=['HHMM'])
    testResultAppend(r.predictions_df_list,predictions_df,r.left_update)

    # We set back values to the original representation:
    test_df_scaled_back = t.svmScaler.inverse_transform(test_df)
    predictions_df_scaled_back = t.svmScaler.inverse_transform(predictions_df)
    y_test_reverse = test_df_scaled_back[:, [1]]
    predictions_reverse = predictions_df_scaled_back[:, [1]]
    # in order to keep real value metrics:
    sv_ratio = clf.support_.shape[0] / t.X_train.size
    testResultAppend(r.sv_ratio_list,sv_ratio,r.left_update)
    errorList = (y_test_reverse - predictions_reverse).flatten()
    testResultAppend(r.std_dev_list,statistics.stdev(errorList),r.left_update)
    testResultAppend(r.error_ds,pd.DataFrame(y_test_reverse - predictions_reverse), r.left_update)
    testResultAppend(r.mean_absolute_errors, metrics.mean_absolute_error(y_test_reverse, predictions_reverse), r.left_update)
    testResultAppend(r.mean_squared_errors, metrics.mean_squared_error(y_test_reverse, predictions_reverse), r.left_update)
    testResultAppend(r.alg_names_list, alg_name, r.left_update)

    # Try to get a more interesting table.
    hhmm_values = np.round(test_df_scaled_back[:, [0]], 0).flatten()
    y_test_reverse = np.round(y_test_reverse, 2).flatten()
    predictions_reverse = np.round(predictions_reverse, 2).flatten()
    errorList = np.round(errorList, 2).flatten()

    value_prediction_error_DF = pd.DataFrame(
    {'HHMM': hhmm_values, 'Real val': y_test_reverse,
     'Prediction': predictions_reverse, "Error": errorList})
    value_prediction_error_DF['Error (%) '] = (value_prediction_error_DF['Error']*100) / value_prediction_error_DF['Real val']
    value_prediction_error_DF = value_prediction_error_DF.sort_values(by=['HHMM'])
    value_prediction_error_DF = value_prediction_error_DF.round(1)
    value_prediction_error_DF['HHMM'] = value_prediction_error_DF['HHMM'].apply(lambda x: datetime.strptime(getTimeString(x), '%H%M.0').time())
    testResultAppend(r.report_df_list,value_prediction_error_DF, r.left_update)


def getTimeString(x):
    original_timeString = str(x)
    timeString = original_timeString
    for count in range(4-len(original_timeString)):
        timeString = "0"+timeString
    return timeString


def processLinearReg(t, r, predictions, alg_name):
    # precision measures and indicators:
    errorList = t.y_test - predictions
    testResultAppend(r.std_dev_list,statistics.stdev(errorList), r.left_update)
    testResultAppend(r.error_ds,pd.DataFrame(errorList), r.left_update)
    testResultAppend(r.mean_absolute_errors, metrics.mean_absolute_error(t.y_test, predictions), r.left_update)
    testResultAppend(r.mean_squared_errors, metrics.mean_squared_error(t.y_test, predictions), r.left_update)
    testResultAppend(r.alg_names_list, alg_name, r.left_update)

    # Try to get a more interesting table.
    hhmm_values = np.round(t.X_test[:,1]).flatten()
    t.y_test = np.round(t.y_test, 2).values.flatten()
    predictions = np.round(predictions, 2).flatten()
    errorList = np.round(errorList, 2).values.flatten()

    value_prediction_error_DF = pd.DataFrame(
        {'HHMM': hhmm_values, 'Real val': t.y_test,
         'Prediction': predictions, "Error": errorList})
    value_prediction_error_DF['Error (%) '] = (value_prediction_error_DF['Error'] * 100) / value_prediction_error_DF[
        'Real val']
    value_prediction_error_DF = value_prediction_error_DF.sort_values(by=['HHMM'])
    value_prediction_error_DF = value_prediction_error_DF.round(1)
    value_prediction_error_DF['HHMM'] = value_prediction_error_DF['HHMM'].apply(
        lambda x: datetime.strptime(getTimeString(x), '%H%M.0').time())
    testResultAppend(r.report_df_list,value_prediction_error_DF, r.left_update)


def getSVMTestData(dataSetDF):
    svmDf = dataSetDF[['HHMM', 'bandwidth']].copy()
    # feature scaling:
    svmScaler = preprocessing.MaxAbsScaler().fit(svmDf)
    svmDf_scaled = svmScaler.transform(svmDf)

    svmDf['HHMM'] = column(svmDf_scaled, 0)
    svmDf['bandwidth'] = column(svmDf_scaled, 1)
    # Selecting predictors and values to predict
    X = svmDf[['HHMM']]
    y = svmDf['bandwidth']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)
    return SVMTestData(X_train, X_test, y_train, y_test, dataSetDF, svmScaler, svmDf_scaled)


def getLinRegTestData(dataSetDF, degree):
    # Selecting predictors and values to predict
    X = dataSetDF[['HHMM']]
    y = dataSetDF['bandwidth']

    poly = PolynomialFeatures(degree=degree)
    X_ = poly.fit_transform(X)

    # train test splitting
    X_train, X_test, y_train, y_test = train_test_split(
        X_, y, test_size=0.4, random_state=101)

    return LinRegTestData(X_train, X_test, y_train, y_test)


def read_data(file_name):
    return pd.read_csv('data/med-output/' + file_name)


def get_predictions_chart(ground_thruth_data, alg_names_list, predictions_df_list):
    series = list()

    ground_thruth = {
        'name': 'Ground Truth',
        'data': ground_thruth_data}

    series.append(ground_thruth)

    for count in range(len(predictions_df_list)):
        series.append(
            {
                'type': 'line',
                'id': alg_names_list[count],
                'name': alg_names_list[count],
                'data': predictions_df_list[count].values.tolist(),
                'lineWidth': 3,
                'marker': {
                    'enabled': False
                },
                'states': {
                    'hover': {
                        'lineWidth': 4
                    }
                },
                'enableMouseTracking': True
            }
        )

    predictionsChart = {
        'chart': {'type': 'scatter', 'zoomType': 'xy'},
        'mapNavigation': {
            'enableMouseWheelZoom': True
        },
        'legend': {'enabled': True},
        'title': {'text': '15 min mean bandwidth usage'},
        'xAxis': {'title': {'enabled': True, 'text': 'Time (HHMM)'},
                  'startOnTick': False,
                  'endOnTick': False,
                  'showLastLabel': True},
        'yAxis': {'title': {'text': 'Usage'}},
        'tooltip': {
            'enabled':True,
            'animation':True,
            'valueSuffix': ''
        },
        'plotOptions': {
            'column': {
                'pointPadding': 0.2,
                'borderWidth': 0
            }
        },
        'series': series,
        'credits': {
            'enabled': False
        }
        }

    predictionsChartDump = json.dumps(predictionsChart, separators=(',', ':'), sort_keys=True, indent=4)
    return predictionsChartDump


def getErrorDistrosChart(predictions_df_list, error_ds, alg_names_list):
    error_series = list()

    for count in range(len(predictions_df_list)):
        error_series.append(
            {
                'type': 'scatter',
                'data': error_ds[count].values.tolist(),
                'id': alg_names_list[count],
                'visible': False,
                'showInLegend': False,
            }
        )
        error_series.append(
            {
                'name': alg_names_list[count],
                'type': 'bellcurve',
                'xAxis': 1,
                'yAxis': 1,
                'baseSeries': alg_names_list[count]
            }
        )

    error_distros_x_axes = list()
    for count in range(len(predictions_df_list) - 1):
        error_distros_x_axes.append(
            {'title': {'text': ''},
             'visible': False,
             }
        )
    error_distros_x_axes.append(
        {'title': {'text': 'Error measures'},
         'visible': True,
         'type': 'linear',
         'tickPosition': 'outside'}
    )

    errorDistrosChart = {
        'chart': {'zoomType': 'x'},
        'legend': {'enabled': True},
        'title': {'text': 'Error distributions'},
        'xAxis': error_distros_x_axes,
        'yAxis': [{'visible': False},
                  {'type': 'linear'}],
        'series': error_series,
        'credits': {
            'enabled': False
        }
    }

    errorDistrosChartDump = json.dumps(errorDistrosChart, separators=(',', ':'), sort_keys=True, indent=4)
    return errorDistrosChartDump


def get_single_errorDistroChart(error_list, alg_name):

    error_series = list()

    error_series.append(
        {
            'name': ' PDF',
            'type': 'bellcurve',
            'intervals': 4,
            'pointsInInterval': 5,
            'xAxis': 1,
            'yAxis': 1,
            'baseSeries': 1,
            'zIndex' : -1,
            'marker': {
                'enabled': 'true'
            }
        }
    )

    error_series.append(
        {
            'name': ' Error measures',
            'type': 'scatter',
            'data': error_list,
            'showInLegend': 'false',
            'marker': {
                'radius': 1.5
            }
        }
    )

    error_distro_x_axes = list()
    error_distro_x_axes.append(
        {'title': {'text': 'Error measure count'},
         'alignTicks': 'false'}
    )

    error_distro_x_axes.append(
        {'title': {'text': 'Error values.'},
        'alignTicks': 'false',
        'opposite': 'true'}
    )

    error_distro_y_axes = list()
    error_distro_y_axes.append(
        {
            'title': {'text': 'Error values'},
            'alignTicks': 'false',
        }
    )
    error_distro_y_axes.append(
        {
            'title': {'text': 'Probability.'},
            'opposite': 'true',
            'alignTicks': 'false'
        }
    )

    single_error_distro_Chart = {
        'chart': {
            'events': {
                'load': ''
            },'zoomType': 'xy'
        },
        'mapNavigation': {
            'enableMouseWheelZoom': 'true'
        },
        'title': {'text': alg_name },
        'xAxis': error_distro_x_axes,
        'yAxis' : error_distro_y_axes,
        'series' : error_series,
        'credits': {
            'enabled': 'false'
        },

    }

    return single_error_distro_Chart


def column(matrix, i):
    return [row[i] for row in matrix]


def getNormalizedMetricsChart(metric_names, r):

    normalized_metric_series = list()
    for count in range(len(metric_names)):
        normalized_metric_series.append(
            {
                'name': metric_names[count],
                'data': r.norm_metrics_df[metric_names[count]].tolist()
            }
        )

    norm_metricsChart = {
        'chart': {'type': 'column', 'zoomType': 'xy'},
        'mapNavigation': {
            'enableMouseWheelZoom': True
        },
        'title': {'text': 'Normalized Metrics'},
        'subtitle': {'text': 'measures taken during test'},

        'xAxis': {'title': {'text': ''},
                  'categories': r.alg_names_list,
                  },
        'yAxis': {
            'min': 0,
            'title': {
                'text': 'Percentage (Normalized)',
                'align': 'high'
            },
            'labels': {
                'overflow': 'justify'
            }
        },
        'tooltip': {
            'valueSuffix': 'X 100%'
        },
        'plotOptions': {
            'bar': {
                'dataLabels': {
                    'enabled': True

                }
            }
        },
        'legend': {
            'enabled': False,
            'layout': 'vertical',
            'align': 'right',
            'verticalAlign': 'top',
            'x': -40,
            'y': 2,
            'floating': True,
            'borderWidth': 1,
            'shadow': True
        },
        'credits': {
            'enabled': False
        },
        'series': normalized_metric_series,

    }

    norm_metricsChartDump = json.dumps(norm_metricsChart, separators=(',', ':'), sort_keys=True, indent=4)
    return norm_metricsChartDump


def getRawMetricsChart(metric_names, abs_metricsDf, alg_names_list):
    raw_metric_series = list()

    for count in range(len(metric_names) - 1):
        raw_metric_series.append(
            {
                'name': metric_names[count],
                'data': abs_metricsDf[metric_names[count]].tolist(),
                'visible': False
            })
        raw_metric_series.append(
            {
                'name': metric_names[count] + "line",
                'showInLegend': False,
                'data': abs_metricsDf[metric_names[count]].tolist(),
                'visible': False,
                'type': 'line'
            })
    raw_metric_series.append(
        {
            'name': metric_names[len(metric_names) - 1],
            'data': abs_metricsDf[metric_names[len(metric_names) - 1]].tolist(),
            'visible': True
        }
    )
    raw_metric_series.append(
        {
            'name': metric_names[len(metric_names) - 1] + "line",
            'showInLegend': False,
            'data': abs_metricsDf[metric_names[len(metric_names) - 1]].tolist(),
            'visible': True,
            'type': 'line'
        })

    rawMetricsChart = {
        'chart': {'type': 'column', 'zoomType': 'xy'},
        'mapNavigation': {
            'enableMouseWheelZoom': True
        },
        'title': {'text': 'Absolute Metrics'},
        'subtitle': {'text': 'measures taken during test'},

        'xAxis': {'title': {'text': ''},
                  'categories': alg_names_list,
                  },
        'yAxis': {
            'min': 0,
            'title': {
                'text': 'Abs values',
                'align': 'high'
            },
            'labels': {
                'overflow': 'justify'
            }
        },
        'tooltip': {
            'valueSuffix': ''
        },
        'plotOptions': {
            'bar': {
                'dataLabels': {
                    'enabled': True

                }
            }
        },
        'legend': {
            'enabled': False,
            'layout': 'vertical',
            'align': 'right',
            'verticalAlign': 'top',
            'x': -40,
            'y': 2,
            'floating': True,
            'borderWidth': 1,
            'shadow': True
        },
        'credits': {
            'enabled': False
        },
        'series': raw_metric_series,

    }

    rawMetricsChartDump = json.dumps(rawMetricsChart, separators=(',', ':'), sort_keys=True, indent=4)
    return rawMetricsChartDump


def getErrorTableContent(df):
    div_table_content = div( 'contentHolder', cls='table-responsive')
    div_table_content.attributes['style']='overflow: auto; height: 400px'

    table_content = df.to_html(index=None)
    table_content = table_content.replace('class="dataframe"', "class='table table-striped' style='text-align: right;'")
    table_content = table_content.replace('border="1"', "")

    div_table_content[0] = table_content

    return div_table_content
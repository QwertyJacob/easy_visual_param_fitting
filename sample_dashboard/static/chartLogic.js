

   // metric_names = ['mean_abs_err', 'mean_sqrd_err', 'std_dev', 'performance (negative)']


let  mean_abs_err_cb;
let  mean_sqrd_err_cb;
let  std_dev_cb;
let  time_cb;
let  sv_ratio_cb;


let  raw_mean_abs_err_cb;
let  raw_mean_sqrd_err_cb;
let  raw_std_dev_cb;
let  raw_time_cb;
let  raw_sv_ratio_cb;

let predictors = ['Polynomial Regression', 'SVR with polynomial Kernel', 'SVR with RBF kernel'];
let params_to_fit = ['deg','C','gamma','epsilon'];

function sleep (time) {
  return new Promise((resolve) => setTimeout(resolve, time));
}

sleep(1000).then(() => {


if(rawMetricChartLoaded){
    raw_mean_abs_err_cb = $('#raw_mean_abs_err_cb');
    raw_mean_sqrd_err_cb = $('#raw_mean_sqrd_err_cb');
    raw_std_dev_cb = $('#raw_std_dev_cb');
    raw_time_cb = $('#raw_time_cb');
    raw_sv_ratio_cb = $('#raw_sv_ratio_cb');
}

if(normalizedMetricChartLoaded){
    mean_abs_err_cb = $('#norm_mean_abs_err_cb');
    mean_sqrd_err_cb = $('#norm_mean_sqrd_err_cb');
    std_dev_cb = $('#norm_std_dev_cb');
    time_cb = $('#norm_time_cb');
    sv_ratio_cb = $('#norm_sv_ratio_cb');
}


raw_mean_abs_err_cb.change(function () {
    var mean_abs_err_series = rawMetricsChart.series[0];
    var mean_abs_err_line = rawMetricsChart.series[1];

    if (mean_abs_err_series.visible) {
        mean_abs_err_series.hide();
        mean_abs_err_line.hide();
    } else {
        mean_abs_err_series.show();
        mean_abs_err_line.show();
    }
});

raw_mean_sqrd_err_cb.change(function () {
    var mean_sqrd_err_series = rawMetricsChart.series[2];
    var mean_sqrd_err_line = rawMetricsChart.series[3];

    if (mean_sqrd_err_series.visible) {
        mean_sqrd_err_series.hide();
        mean_sqrd_err_line.hide();
    } else {
        mean_sqrd_err_series.show();
        mean_sqrd_err_line.show();
    }
});

raw_std_dev_cb.change(function () {
    var std_dev_series = rawMetricsChart.series[4];
    var std_dev_line = rawMetricsChart.series[5];

    if (std_dev_series.visible) {
        std_dev_series.hide();
        std_dev_line.hide();
    } else {
        std_dev_series.show();
        std_dev_line.show();
    }
});

raw_time_cb.change(function () {
    var time_series = rawMetricsChart.series[6];
    var time_line = rawMetricsChart.series[7];

    if (time_series.visible) {
        time_series.hide();
        time_line.hide();
    } else {
        time_series.show();
        time_line.show();
    }
});

raw_sv_ratio_cb.change(function () {
    var sv_ratio_series = rawMetricsChart.series[8];
    var sv_ratio_line = rawMetricsChart.series[9];

    if (sv_ratio_series.visible) {
        sv_ratio_series.hide();
        sv_ratio_line.hide();
    } else {
        sv_ratio_series.show();
        sv_ratio_line.show();
    }
});


mean_abs_err_cb.change(function () {
    var mean_abs_err_series = normalizedMetricsChart.series[0];

    if (mean_abs_err_series.visible) {
        mean_abs_err_series.hide();
    } else {
        mean_abs_err_series.show();
    }
});

mean_sqrd_err_cb.change(function () {
    var mean_sqrd_err_series = normalizedMetricsChart.series[1];

    if (mean_sqrd_err_series.visible) {
        mean_sqrd_err_series.hide();
    } else {
        mean_sqrd_err_series.show();
    }
});

std_dev_cb.change(function () {
    var std_dev_series = normalizedMetricsChart.series[2];

    if (std_dev_series.visible) {
        std_dev_series.hide();
    } else {
        std_dev_series.show();
    }
});

time_cb.change(function () {
    var time_series = normalizedMetricsChart.series[3];

    if (time_series.visible) {
        time_series.hide();
    } else {
        time_series.show();
    }
});

sv_ratio_cb.change(function () {
    var sv_ratio_series = normalizedMetricsChart.series[4];

    if (sv_ratio_series.visible) {
        sv_ratio_series.hide();
    } else {
        sv_ratio_series.show();
    }
});

let errorChartsEventsLoad = function () {
                Highcharts.each(this.series[0].data, function (point, i) {
                    var labels = ['4σ', '3σ', '2σ', 'σ', 'μ', 'σ', '2σ', '3σ', '4σ'];
                    if (i % 5 === 0) {
                        point.update({
                            color: 'black',
                            dataLabels: {
                                enabled: true,
                                format: labels[Math.floor(i / 5)],
                                overflow: 'none',
                                crop: false,
                                y: -2,
                                style: {
                                    fontSize: '13px'
                                }
                            }
                        });
                    }
                });
            };

let addCumulative = function(chart){
    cumulativeData = [0];
    primalData = chart.series[1]['data'];
    primalData.forEach(function(elementToAdd, index)
    {
        var newElement = cumulativeData[index] + elementToAdd;
        cumulativeData.push(newElement);
    });
    chart.series.push({
        'type': 'line',
        'data': cumulativeData.shift()
    });
};

if (errorChartsLoaded){
    for (i = 0; i < errorCharts.length; i++) {
        addCumulative(errorCharts[i]);
        errorCharts[i].chart.events.load = errorChartsEventsLoad;
        errorCharts[i].credits.enabled = false;
        errorCharts[i].series[2].showInLegend = false;
        errorCharts[i].series[2].visible = false;
         Highcharts.chart(algorithm_names[i]+'-chart-div', errorCharts[i]);
    }
}

var series_colors = ['#2f7ed8', '#0d233a', '#8bbc21', '#910000', '#1aadce',
    '#492970', '#f28f43', '#77a1e5', '#c42525', '#a6c96a'];

/*
CB_Label = $('#ground_truth_CB_Label');
CB_Label[0].style.color = "white";
CB_Label[0].style.background = series_colors[0];
CB_Span = $("#ground_truth_CB_Span");
CB_Span[0].style.color = series_colors[0];
CB_Span[0].style.background ="white";
CB = $("#ground_truth_CB");
CB[0].checked = true;
CB.change(function () {
    var current_series = predictionsChart.series[0];
    if (current_series.visible) {
        current_series.hide();
    } else {
        current_series.show();
    }
});

 for (let i = 1; i <= algoritm_names.length; i++) {
    CB_Label = $('#'+algoritm_names[i-1]+"_CB_Label");
    CB_Label[0].style.color = "white";
    CB_Label[0].style.background = series_colors[i];
    CB_Span = $('#'+algoritm_names[i-1]+"_CB_Span");
    CB_Span[0].style.color = series_colors[i];
    CB_Span[0].style.background ="white";
    CB = $('#'+algoritm_names[i-1]+"_CB");
    CB[0].checked = true;
    CB.change(function () {
        var current_series = predictionsChart.series[i];
        if (current_series.visible) {
            current_series.hide();
        } else {
            current_series.show();
        }
    });
    }
*/

    let update_error_charts = function(data){
        let errorCharts = data.errorCharts;
        for ( i = 0; i < errorCharts.length; i++) {
            addCumulative(errorCharts[i]);
            errorCharts[i].chart.events.load = errorChartsEventsLoad;
            errorCharts[i].credits.enabled = false;
            errorCharts[i].series[2].showInLegend = false;
            errorCharts[i].series[2].visible = false;
            Highcharts.chart(data.alg_names[i]+'-chart-div', errorCharts[i]);
        }
    };

    let update_error_tables = function(data){
      document.getElementById("table_data_div").innerHTML = data.table_data;
    };

    let update_error_div = function(data){

        update_error_tables(data);

        update_error_charts(data);



    };

    let cleanMetricsCharts = function () {
        while (normalizedMetricsChart.series.length > 0)
            normalizedMetricsChart.series[0].remove(true);
        while (rawMetricsChart.series.length > 0)
            rawMetricsChart.series[0].remove(true);
    };

    let resetMetricButtons = function(){

        raw_mean_abs_err_cb.prop("checked", true);
        raw_mean_sqrd_err_cb.prop("checked", true);
        raw_std_dev_cb.prop("checked", true);
        raw_time_cb.prop("checked", true);
        raw_sv_ratio_cb.prop("checked", true);
        mean_abs_err_cb.prop("checked", true);
        mean_sqrd_err_cb.prop("checked", true);
        std_dev_cb.prop("checked", true);
        time_cb.prop("checked", true);
        sv_ratio_cb.prop("checked", true);

    };

    let updateMetricsCharts = function(data){

         for(norm_metric_series of data.normMetricsToAppend){
            normalizedMetricsChart.addSeries(norm_metric_series)
        }

        for(raw_metric_series of data.rawMetricsToAppend){
            rawMetricsChart.addSeries(raw_metric_series)
        }

        normalizedMetricsChart.xAxis[0].setCategories(data.alg_names_list);
        rawMetricsChart.xAxis[0].setCategories(data.alg_names_list);
    };

    let increase_report = function(data){

        for(prediction_series of data.predictionsToAppend){
            predictionsChart.addSeries(prediction_series)
        }

        cleanMetricsCharts();

        updateMetricsCharts(data);

        resetMetricButtons();

        update_error_div(data);
    };

    let decrease_report = function(data){
        for(alg_name_to_remove of data.alg_names_to_remove){
            predictionsChart.get(alg_name_to_remove).remove();
        }

        cleanMetricsCharts();

        updateMetricsCharts(data);

        update_error_div(data);

    };


    let paramSlider = $("#param_slider");

    class SliderSettings{
        constructor(min_value, max_value, step_value, current_from_value, current_to_value){
            this.min = min_value;
            this.max = max_value;
            this.step = step_value;
            this.from = current_from_value;
            this.to = current_to_value;
        }
    }

  paramSlider.ionRangeSlider({
        type: "double",
        grid: true,
        prefix: param_to_fit+" = ",
        onFinish: function (data) {
            predictionsChart.showLoading();
            rawMetricsChart.showLoading();
            normalizedMetricsChart.showLoading();

             $.ajax({
                url: '/ajax/update_report/',
                data: {
                    'to': data.to,
                    'from': data.from,
                    'predictor' : predictor,
                    'param_to_fit' : param_to_fit
                },
                dataType: 'json',

                success: function (data) {

                    if(data.increase) increase_report(data);
                    else decrease_report(data);
                    predictionsChart.hideLoading();
                    rawMetricsChart.hideLoading();
                    normalizedMetricsChart.hideLoading();
                }
              });

        }

    });

   let paramSliderOptions = paramSlider.data("ionRangeSlider");

    let linearSS = new SliderSettings(1,20,1,4,7);
    let polysvrdegSS = new SliderSettings(1,20,1,4,7);
    let polysvrgammaSS = new SliderSettings(4.0,6.0,0.1,5.0,5.2);
    let polysvrepsilonSS = new SliderSettings(0.05,0.15,0.005,0.085,0.1);
    let polysvrCSS = new SliderSettings(1,100,5,1,21);
    let rbfsvrgammaSS = new SliderSettings(5.1,5.3,0.01,5.18,5.21);
    let rbfsvrepsilonSS = new SliderSettings(0.01,0.2,0.03,0.03,0.12);
    let rbfsvrCSS = new SliderSettings(0.1,10,0.5,0.1,1.5);

    let updateParamSliderOptions = function(){
        switch (predictor) {
            case predictors[0]:
                paramSliderOptions.update({from:linearSS.from, to: linearSS.to,
                    step:linearSS.step, min: linearSS.min, max:linearSS.max});
                break;
            case predictors[1]:
                switch (param_to_fit) {
                    case params_to_fit[0]://deg
                        paramSliderOptions.update({from:polysvrdegSS.from,
                                                    to: polysvrdegSS.to, step:polysvrdegSS.step,
                                                    min: polysvrdegSS.min, max:polysvrdegSS.max});
                        break;
                    case params_to_fit[1]://C
                        paramSliderOptions.update({from:polysvrCSS.from,
                                                    to: polysvrCSS.to, step:polysvrCSS.step,
                                                    min: polysvrCSS.min, max:polysvrCSS.max});
                        break;
                    case params_to_fit[2]://gamma
                        paramSliderOptions.update({from:polysvrgammaSS.from,
                                                    to: polysvrgammaSS.to, step:polysvrgammaSS.step,
                                                    min: polysvrgammaSS.min, max:polysvrgammaSS.max});
                        break;
                    case params_to_fit[3]://epsilon
                        paramSliderOptions.update({from:polysvrepsilonSS.from,
                                                    to: polysvrepsilonSS.to, step:polysvrepsilonSS.step,
                                                    min: polysvrepsilonSS.min, max:polysvrepsilonSS.max});
                        break;
                }
                break;
            case predictors[2]:
                 switch (param_to_fit) {
                    case params_to_fit[1]://C
                         paramSliderOptions.update({from:rbfsvrCSS.from,
                                                    to: rbfsvrCSS.to, step:rbfsvrCSS.step,
                                                    min: rbfsvrCSS.min, max:rbfsvrCSS.max});
                        break;
                    case params_to_fit[2]://gamma
                        paramSliderOptions.update({from:rbfsvrgammaSS.from,
                                                    to: rbfsvrgammaSS.to, step:rbfsvrgammaSS.step,
                                                    min: rbfsvrgammaSS.min, max:rbfsvrgammaSS.max});
                        break;
                    case params_to_fit[3]://epsilon
                        paramSliderOptions.update({from:rbfsvrepsilonSS.from,
                                                    to: rbfsvrepsilonSS.to, step:rbfsvrepsilonSS.step,
                                                    min: rbfsvrepsilonSS.min, max:rbfsvrepsilonSS.max});
                        break;
                }
                break;
        }
    };

    updateParamSliderOptions();
    let degree_slider_loader = $('#degree_slider_loader');
    degree_slider_loader.hide();
    paramSlider.show();


});




"""dashboard URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from django.conf.urls import url

from sample_dashboard import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('index/', views.index, name='index'),
    path('dashboard', views.dashboard, name='dashboard'),
    path('rbf_svr_gamma', views.rbf_svr_gamma, name='rbf_svr_gamma'),
    path('rbf_svr_epsilon', views.rbf_svr_epsilon, name='rbf_svr_epsilon'),
    path('rbf_svr_C', views.rbf_svr_C, name='rbf_svr_C'),
    path('poly_reg', views.poly_reg, name='poly_reg'),
    url('evaluate/$', views.evaluate, name='evaluate'),
    url('update_evaluation/$', views.update_evaluation, name='update_evaluation'),
    path('poly_svr_deg', views.poly_svr_deg, name='poly_svr_deg'),
    path('poly_svr_gamma', views.poly_svr_gamma, name='poly_svr_gamma'),
    path('poly_svr_epsilon', views.poly_svr_epsilon, name='poly_svr_epsilon'),
    path('poly_svr_C', views.poly_svr_C, name='poly_svr_C'),
    url(r'^ajax/update_report/$', views.update_report, name='update_report'),

]


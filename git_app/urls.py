from django.urls import path
from . import views

urlpatterns = [
    path('weekly_report/', views.get_weekly_report, name='weekly_report'),

]
from django.urls import path
from . import views

app_name = 'iris'

urlpatterns = [
    path('predict/', views.iris_prediction, name='predict'),
]

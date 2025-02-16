from django.urls import path
from . import views

app_name = 'pytorch'

urlpatterns = [
    path('', views.index, name='index'),
    path('house-prediction/', views.house_prediction, name='house_prediction'),
    path('prediction-result/<int:pk>/', views.prediction_result, name='prediction_result'),
] 
from django.urls import path
from . import views

app_name = 'housepriceprediction'

urlpatterns = [
    path('house-price-prediction/', views.index, name='index'),
    path('prediction-result/<int:pk>/', views.prediction_result, name='prediction_result'),
] 
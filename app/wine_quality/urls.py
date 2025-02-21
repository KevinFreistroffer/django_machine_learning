from django.urls import path
from . import views

app_name = 'wine_quality'
urlpatterns = [
    path('', views.wine_quality_view, name='predict'),
] 
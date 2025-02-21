from django.urls import path
from . import views

urlpatterns = [
    path('', views.wine_quality_view, name='wine_quality'),
] 
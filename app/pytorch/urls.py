from django.urls import path
from . import views

app_name = 'pytorch'

urlpatterns = [
    path('', views.index, name='index'),
] 
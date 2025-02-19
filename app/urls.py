from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('iris/', include('app.iris.urls')),
    path('house-price-prediction/', include('app.housepriceprediction.urls')),
] 
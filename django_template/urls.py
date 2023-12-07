from django.contrib import admin 
from django.urls import path 
from prediction_app.views import predict_price 


urlpatterns = [ 
    path('admin/', admin.site.urls), 
    path('predict/', predict_price), 
] 
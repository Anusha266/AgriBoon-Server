from .views import PricePredictionAPIView
from django.urls import path

app_name = 'api'

urlpatterns = [
    path('fetch_price/', PricePredictionAPIView.as_view(), name='fetch_price'),
]

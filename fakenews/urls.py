from django.urls import path
from . import views

urlpatterns = [
    path('', views.home_view, name='home'),
    path('api/reddit/<str:subreddit>/', views.get_reddit_data, name='reddit_data'),
    path('api/verify-news/', views.verify_news, name='verify_news'),
]


from home import views
from django.urls import path

urlpatterns = [
    path("",views.home,name = "home"),
    path("prediction",views.prediction,name = "prediction"),
    path("register-user",views.registerView,name = "register-user"),
    path("login-user",views.loginView,name = "login-user"),
    path("dataset",views.view_dataset,name = "dataset"),
    path("train_model",views.train_model,name = "train_model"),
    path('user-logout',views.userlogout,name='user-logout'),
]

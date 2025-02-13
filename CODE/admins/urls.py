from admins import views
from django.urls import path

urlpatterns = [
    path('admin-login',views.adminlogin,name='admin-login'),
    path('admin-home',views.adminhome,name='admin-home'),
    path('userslist',views.userslist,name='userslist'),
    path('admin-logout',views.adminlogout,name='admin-logout'),

]


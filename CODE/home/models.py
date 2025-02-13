from django.contrib.auth.models import User
from django.db import models 

class Profile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    mobile = models.CharField(max_length=15)
    locality = models.CharField(max_length=100)
    address = models.TextField()

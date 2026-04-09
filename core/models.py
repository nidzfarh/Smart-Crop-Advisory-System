from django.db import models

from django.contrib.auth.models import User

class SeedShop(models.Model):
    name = models.CharField(max_length=100)
    location = models.CharField(max_length=100)
    district = models.CharField(max_length=100, default='Kerala')
    crop_available = models.CharField(max_length=100)
    price_range = models.CharField(max_length=50) # Low, Medium, High
    contact_info = models.CharField(max_length=100, blank=True)
    # Simple coordinates for distance calculation (Mock values for now)
    latitude = models.FloatField(default=0.0)
    longitude = models.FloatField(default=0.0)

    def __str__(self):
        return f"{self.name} ({self.district})"

class UserCrop(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    crop_name = models.CharField(max_length=100)
    planting_date = models.DateField()
    status = models.CharField(max_length=50, default='Growing') # Growing, Harvested
    
    def __str__(self):
        return f"{self.user.username} - {self.crop_name}"

class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    district = models.CharField(max_length=100)

    def __str__(self):
        return f"{self.user.username} - {self.district}"

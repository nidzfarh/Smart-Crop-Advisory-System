from django import forms
from django.contrib.auth.models import User
from .models import SeedShop, UserCrop
from .weather_utils import KERALA_DISTRICT_CLIMATE

class UserRegisterForm(forms.ModelForm):
    password = forms.CharField(widget=forms.PasswordInput)
    confirm_password = forms.CharField(widget=forms.PasswordInput)
    district = forms.ChoiceField(
        choices=[(d, d) for d in KERALA_DISTRICT_CLIMATE.keys()],
        label="District (Kerala)",
        widget=forms.Select(attrs={'class': 'form-select'})
    )

    class Meta:
        model = User
        fields = ['username', 'email', 'password', 'district']

    def clean(self):
        cleaned_data = super().clean()
        password = cleaned_data.get("password")
        confirm_password = cleaned_data.get("confirm_password")

        if password != confirm_password:
            raise forms.ValidationError("Passwords do not match")
        return cleaned_data

class CropTrackingForm(forms.ModelForm):
    planting_date = forms.DateField(widget=forms.DateInput(attrs={'type': 'date'}))
    class Meta:
        model = UserCrop
        fields = ['crop_name', 'planting_date', 'status']

class FarmDataInputForm(forms.Form):
    nitrogen = forms.IntegerField(min_value=0, max_value=200, label="Nitrogen (N)")
    phosphorus = forms.IntegerField(min_value=0, max_value=200, label="Phosphorus (P)")
    potassium = forms.IntegerField(min_value=0, max_value=200, label="Potassium (K)")
    ph = forms.FloatField(min_value=0.0, max_value=14.0, label="Soil pH")
    
    location = forms.ChoiceField(
        choices=[(d, d) for d in KERALA_DISTRICT_CLIMATE.keys()], 
        initial='Malappuram', 
        widget=forms.Select(attrs={'class': 'form-select'})
    )
    
    soil_type = forms.ChoiceField(
        choices=[(s, s) for s in ['Laterite', 'Sandy', 'Clayey', 'Loamy', 'Alluvial', 'Red', 'Forest']],
        initial='Laterite',
        label='Soil Type',
        widget=forms.Select(attrs={'class': 'form-select'})
    )
    
    # Farm Size Planner (Optional)
    land_area_value = forms.DecimalField(
        required=False, 
        min_value=0, 
        max_digits=10, 
        decimal_places=2, 
        label="Farm Size", 
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Optional (e.g. 0.5)'})
    )
    land_area_unit = forms.ChoiceField(
        required=False, 
        choices=[('Acre', 'Acre'), ('Cent', 'Cent'), ('Hectare', 'Hectare')], 
        initial='Acre', 
        widget=forms.Select(attrs={'class': 'form-select'})
    )
    
    crop_name = forms.CharField(required=False, label="Preferred Crop (Optional)")

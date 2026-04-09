from django.test import TestCase, Client
from django.urls import reverse
from django.contrib.auth.models import User
from core.models import UserCrop

class UIIntegrityTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_user(username='testuser', password='password123')
        self.client.login(username='testuser', password='password123')

    def test_predict_page_status(self):
        """Verify that predict.html renders successfully."""
        response = self.client.get(reverse('predict'))
        self.assertEqual(response.status_code, 200)
        
        # Check for Loading Overlay
        self.assertContains(response, 'id="loadingOverlay"')
        self.assertContains(response, 'Executing Intelligence Engine')

    def test_add_crop_page_status(self):
        """Verify that add_crop.html renders successfully."""
        response = self.client.get(reverse('add_crop'))
        self.assertEqual(response.status_code, 200)

    def test_result_page_logic_and_tags(self):
        """Verify that result page renders correctly without literal template tags."""
        # Post valid data to trigger result calculation
        post_data = {
            'nitrogen': 50,
            'phosphorus': 50,
            'potassium': 50,
            'ph': 6.5,
            'location': 'Kasaragod',
            'land_area_value': 1,
            'land_area_unit': 'Acre'
        }
        response = self.client.post(reverse('result'), post_data)
        self.assertEqual(response.status_code, 200)
        
        # Verify rendered content
        self.assertContains(response, 'Analysis Report')
        self.assertContains(response, 'Market Analysis')

    def test_dashboard_functionality(self):
        """Ensure dashboard renders."""
        response = self.client.get(reverse('dashboard'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'My Farm Status')

class AuthUITests(TestCase):
    def setUp(self):
        self.client = Client()

    def test_login_page_wording(self):
        """Verify simplified wording on login page."""
        response = self.client.get(reverse('login'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Sign In')
        self.assertContains(response, 'Username')
        self.assertContains(response, 'Password')
        self.assertNotContains(response, 'Digital Identity')
        self.assertNotContains(response, 'Encrypted Key')

    def test_register_page_wording(self):
        """Verify simplified wording on registration page."""
        response = self.client.get(reverse('register'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Create Account')
        self.assertContains(response, 'Email Address')
        self.assertNotContains(response, 'Intelligence Profile')
        self.assertNotContains(response, 'Communication Channel')

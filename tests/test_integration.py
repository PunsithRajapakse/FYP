import unittest
from app import app

class TestIntegration(unittest.TestCase):
    def setUp(self):
        """Set up test client"""
        self.client = app.test_client()
        self.client.testing = True

    def test_homepage_loads(self):
        """Check if homepage loads successfully"""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)

    def test_file_upload(self):
        """Test if file upload and calorie estimation work"""
        with open("static/uploads/lemon.jpg", "rb") as test_image:
            response = self.client.post("/", data={"file": test_image}, content_type="multipart/form-data")
        
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Estimated Calories", response.data)

if __name__ == "__main__":
    unittest.main()

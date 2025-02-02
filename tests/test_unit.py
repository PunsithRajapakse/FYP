import unittest
from app import classify_food, calculate_calories

class TestUnit(unittest.TestCase):
    def test_classify_food(self):
        """Test if the model correctly classifies a food type"""
        food = classify_food("static/uploads/lemon.jpg")
        self.assertEqual(food, "lemon")  # Expected result

    def test_calculate_calories(self):
        """Test if calorie calculation is correct"""
        weight, calories = calculate_calories("lemon", 4)
        self.assertAlmostEqual(calories, 71.52, places=2)  # Approximate expected value

if __name__ == "__main__":
    unittest.main()

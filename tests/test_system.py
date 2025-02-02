from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import time

# Set up Selenium WebDriver
driver = webdriver.Chrome()  # Ensure ChromeDriver is installed
driver.get("http://127.0.0.1:5000")  # Start Flask app before testing

# Upload an image
upload = driver.find_element(By.NAME, "file")
upload.send_keys("D:/IIT/4th Year/FYP/IPD/Model/static/uploads/lemon.jpg")

# Submit the form
submit_button = driver.find_element(By.TAG_NAME, "button")
submit_button.click()

# Wait for results to load
time.sleep(3)

# Check if results appear
assert "Estimated Calories" in driver.page_source
print("✅ System Test Passed!")

driver.quit()

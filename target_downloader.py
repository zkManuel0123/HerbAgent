from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import os
import time

def download_syndrome_targets(syndrome_id: str, download_dir: str = "downloads"):
    """
    Download target files from SymMap website using Selenium
    
    Args:
        syndrome_id: The syndrome ID (e.g., "00212")
        download_dir: Directory to save downloaded files
    """
    # Create downloads directory if it doesn't exist
    download_dir = os.path.abspath(download_dir)
    os.makedirs(download_dir, exist_ok=True)
    
    # Configure Chrome options
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_experimental_option(
        "prefs",
        {
            "download.default_directory": download_dir,
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "safebrowsing.enabled": True
        }
    )
    
    try:
        # Initialize the driver
        driver = webdriver.Chrome(options=chrome_options)
        
        # Navigate to the page
        url = f"http://www.symmap.org/detail/SMSY{syndrome_id:05d}"
        print(f"Navigating to {url}")
        driver.get(url)
        
        # Wait for the Target button and click it
        target_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//div[@id='button_select_group']//button[text()='Target']"))
        )
        print("Clicking Target button")
        target_button.click()
        
        # Wait for data to load
        time.sleep(2)  # Give time for JavaScript to update the content
        
        # Wait for the Download button and click it
        download_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//*[@id='dl-btn']"))
        )
        print("Clicking Download button")
        download_button.click()

        
        
        # Wait for download to complete
        time.sleep(3)  # Adjust this time based on file size and network speed
        
        print(f"Download completed to {download_dir}")
        
    except TimeoutException as e:
        print(f"Timeout error: {e}")
    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        driver.quit()

if __name__ == "__main__":
    # Test the function with syndrome ID "212"
    SYNDROME_ID = "212"
    download_syndrome_targets(SYNDROME_ID)
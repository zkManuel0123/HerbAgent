from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import os
import time

def download_syndrome_targets(syndrome_ids: list[str], download_dir: str = "downloads"):
    """
    Download target files from SymMap website using Selenium for multiple syndromes
    
    Args:
        syndrome_ids: List of syndrome IDs (e.g., ["00212", "00213"])
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
        driver = webdriver.Chrome(options=chrome_options)
        
        for syndrome_id in syndrome_ids:
            try:
                # Navigate to the page
                url = f"http://www.symmap.org/detail/SMSY{syndrome_id.zfill(5)}"
                print(f"\nProcessing syndrome {syndrome_id}")
                print(f"Navigating to {url}")
                driver.get(url)
                
                # Wait for the Target button and click it
                target_button = WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((By.XPATH, "//div[@id='button_select_group']//button[text()='Target']"))
                )
                print("Clicking Target button")
                target_button.click()
                
                # Wait for data to load
                time.sleep(3)
                
                # Wait for the Download button and click it
                download_button = WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((By.XPATH, "//*[@id='dl-btn']"))
                )
                print("Clicking Download button")
                download_button.click()
                
                # Wait for download to complete
                time.sleep(3)
                
                # 获取下载文件的默认名称
                default_filename = "data.csv"  # SymMap网站下载的默认文件名
                downloaded_file = os.path.join(download_dir, default_filename)
                
                # 构建新的文件名（添加syndrome_id）
                new_filename = f"data_{syndrome_id}.csv"
                new_file_path = os.path.join(download_dir, new_filename)
                
                # 如果目标文件已存在，先删除
                if os.path.exists(new_file_path):
                    os.remove(new_file_path)
                
                # 重命名文件
                if os.path.exists(downloaded_file):
                    os.rename(downloaded_file, new_file_path)
                    print(f"File renamed to: {new_filename}")
                else:
                    print(f"Warning: Downloaded file not found at {downloaded_file}")
                
                print(f"Download completed for syndrome {syndrome_id}")
                
            except TimeoutException as e:
                print(f"Timeout error for syndrome {syndrome_id}: {e}")
                continue
            except Exception as e:
                print(f"Error occurred for syndrome {syndrome_id}: {e}")
                continue
                
    finally:
        driver.quit()

if __name__ == "__main__":
    # Test the function with multiple syndrome IDs
    SYNDROME_IDS = ["212", "001", "100"]
    download_syndrome_targets(SYNDROME_IDS)
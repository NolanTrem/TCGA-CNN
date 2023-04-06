from selenium import webdriver

# Creates a selenium webdriver to scrape the page in Chrome
#options = webdriver.ChromeOptions()
#options.add_argument("--headless")
driver = webdriver.Chrome()

driver.get("https://portal.gdc.cancer.gov/files/889789cf-b211-4788-b62e-1c6be4dba889")
if "Primary Tumor" in driver.page_source:
    print('Found it!')
else if ""
else:
    print('Did not find it.')
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
import sys

driver = webdriver.Chrome()
filename = sys.argv[1]

with open(filename, mode='r') as file:
    lines = file.readlines()

    header = lines.pop(0)

    with open(filename, mode='w') as new_file:
        new_file.write(header)

        for line in lines:
            # Split the line on tabs
            entries = line.strip().split('\t')

            # Check if there are at least two entries in the line
            if len(entries) >= 2:
                # Reset the web driver to a new page
                driver.get("about:blank")

                # Navigate to the URL for the current entry
                driver.get("https://portal.gdc.cancer.gov/files/" + entries[0])

                wait = WebDriverWait(driver, 2)

                try:
                    element = wait.until(
                        EC.presence_of_element_located((By.XPATH, "//*[contains(text(), 'Primary Tumor') "
                                                                  "or contains(text(), "
                                                                  "'Solid Tissue Normal')]")))

                    if "Primary Tumor" in driver.page_source:
                        entries.append("Primary Tumor")
                    elif "Solid Tissue Normal" in driver.page_source:
                        entries.append("Solid Tissue Normal")
                    else:
                        entries.append("Not found")
                except TimeoutException:
                    entries.append("Error")

                # Join the line back together with tabs
                new_line = "\t".join(entries) + "\n"
                new_file.write(new_line)

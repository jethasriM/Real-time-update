import requests
from bs4 import BeautifulSoup

def check_product_availability(product_url):
    try:
        response = requests.get(product_url)
        response.raise_for_status()  # Check for HTTP request errors

        soup = BeautifulSoup(response.text, 'html.parser')

        # Adjust the selector based on the Walmart page structure
        availability_element = soup.find('div', class_='prod-AvailabilityStatus')
        
        if availability_element:
            availability_text = availability_element.text.strip()
            return availability_text
        else:
            return 'Availability information not found'
    except requests.RequestException as e:
        return f'Error: {e}'

# Example usage
product_url = 'https://www.walmart.com/ip/Samsung-Galaxy-S21-Ultra-5G-Smartphone-128GB/641645144'
availability = check_product_availability(product_url)
print(availability)

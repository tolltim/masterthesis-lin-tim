import requests
import datetime
import os
import time

#same as coordinates.csv, check the number there

coordinates = [
    "17436,11372",
    "17437,11372",
    "17438,11372",
    "17439,11372",
    "17440,11372",
    "17436,11373",
    "17437,11373",
    "17438,11373",
    "17439,11373",
    "17440,11373",
    "17436,11374",
    "17437,11374",
    "17438,11374",
    "17439,11374",
    "17440,11374",
    "17436,11375",
    "17437,11375",
    "17438,11375",
    "17439,11375",
    "17440,11375",
    "17436,11376",
    "17437,11376",
    "17438,11376",
    "17439,11376",
    "17440,11376",
    "17436,11377",
    "17437,11377",
    "17438,11377",
    "17439,11377",
    "17440,11377"
]
### Install add in to get tms url to get the pictures: Search for JOSM Strava Heatmap in chrome webstore and fill in template
url_template = ""
def login_to_strava():
    """
    Yoo here you need your credentials of your strava account
    """
    login_url = "https://www.strava.com/login"
    username = ""
    password = ""

    # Create a session
    session = requests.Session()

    # Perform login
    login_data = {
        "email": username,
        "password": password,
        "submit": "Log in"
    }
    session.post(login_url, data=login_data)

    return session

# Function to save PNG image to the output folder
def save_image(url, output_folder, session, image_id):
    response = session.get(url)
    if response.status_code == 200:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d")
        filename = os.path.join(output_folder, f"{timestamp}_{image_id}.png")
        with open(filename, "wb") as f:
            f.write(response.content)
        print(f"Image saved: {filename}")
    else:
        print(f"Failed to download image from URL: {url}")

# Main function
def main():
    output_folder = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_folder, exist_ok=True)

    session = login_to_strava()  # Login to Strava before opening URLs

    for i, coordinate in enumerate(coordinates, start=1):
        lat, lng = coordinate.split(",")
        url = url_template.format(lat=lat.strip(), lng=lng.strip())
        image_id = f"image_{i}"
        save_image(url, output_folder, session, image_id)

if __name__ == "__main__":
    main()
    print("Good Job you idiot horse")
    time.sleep(3)

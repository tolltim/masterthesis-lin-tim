import json
import os
from PIL import Image
import glob
import math


# Directory containing the PNG images
image_directory = "output"

json_directory = "color_codes"


def calculate_brightness(rgb):
    r, g, b = rgb
    brightness = math.sqrt(0.299 * r**2 + 0.587 * g**2 + 0.114 * b**2)
    return brightness


def save_color_codes(image_path, json_path):
    image = Image.open(image_path)

    # Convert the image to RGBA mode if it has transparency
    if image.mode == "P" and "transparency" in image.info:
        image = image.convert("RGBA")

    # Convert the image to RGB mode if it's not already
    if image.mode != "RGB":
        image = image.convert("RGB")

    width, height = image.size
    brightness_value = []

    # Iterate over each pixel and store its color code and brightness
    for y in range(height):
        for x in range(width):
            r, g, b = image.getpixel((x, y))
            brightness = calculate_brightness((r, g, b))
            rounded_brightness = round(brightness, 2)
            brightness_value.append(rounded_brightness)


    with open(json_path, "w") as file:
        json.dump(brightness_value, file)





if not os.path.exists(json_directory):
    os.makedirs(json_directory)


image_paths = glob.glob(os.path.join(image_directory, "*.png"))


for image_path in image_paths:
    # Generate the JSON file path based on the image name
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    json_path = os.path.join(json_directory, f"{image_name}.json")


    save_color_codes(image_path, json_path)

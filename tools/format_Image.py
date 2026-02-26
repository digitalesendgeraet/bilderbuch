from PIL import Image
import os

# Bild laden
n = 0
for images in os.listdir("real_images"):
    img = Image.open("real_images/" + images)

    # Bild auf 640 264
    cropped_img = img.crop((188, 0, 452, 264))
    resized_img = cropped_img.resize((100, 100))
    gray_imag = resized_img.convert("L")
    gray_imag.save("formated_images/test_"+ str(n) +".png")
    n += 1
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
    
    img = Image.open("real_images/" + images)
    # Bild auf 640 264
    cropped_img = img.crop((183, 0, 447, 264))
    resized_img = cropped_img.resize((100, 100))
    gray_imag = resized_img.convert("L")
    gray_imag.save("formated_images2/test_"+ str(n) +"_left.png")

    img = Image.open("real_images/" + images)
    # Bild auf 640 264
    cropped_img = img.crop((183, 0, 447, 264))
    resized_img = cropped_img.resize((100, 100))
    gray_imag = resized_img.convert("L")
    gray_imag.save("formated_images2/test_"+ str(n) +"_right.png")
    n += 1

for images in os.listdir("real_images"):
    img = Image.open("real_images/" + images)

    # Bild auf 640 264
    cropped_img = img.crop((0, 0, 264, 264))
    resized_img = cropped_img.resize((100, 100))
    gray_imag = resized_img.convert("L")
    gray_imag.save("formated_images/test_"+ str(n) +".png")
    

    img = Image.open("real_images/" + images)
    # Bild auf 640 264
    cropped_img = img.crop((0, 0, 269, 264))
    resized_img = cropped_img.resize((100, 100))
    gray_imag = resized_img.convert("L")
    gray_imag.save("formated_images2/test_"+ str(n) +"_right.png")

    n += 1

for images in os.listdir("real_images"):
    img = Image.open("real_images/" + images)

    # Bild auf 640 264
    cropped_img = img.crop((376, 0, 640, 264))
    resized_img = cropped_img.resize((100, 100))
    gray_imag = resized_img.convert("L")
    gray_imag.save("formated_images/test_"+ str(n) +".png")
    

    img = Image.open("real_images/" + images)
    # Bild auf 640 264
    cropped_img = img.crop((371, 0, 635, 264))
    resized_img = cropped_img.resize((100, 100))
    gray_imag = resized_img.convert("L")
    gray_imag.save("formated_images2/test_"+ str(n) +"_left.png")

    n += 1
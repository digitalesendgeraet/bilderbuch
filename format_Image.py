from PIL import Image

# Bild laden
img = Image.open('testImage.jpg')

# Bild auf 640 264
cropped_img = img.crop((188, 0, 452, 264))
resized_img = cropped_img.resize((100, 100))
gray_imag = resized_img.convert("L")
gray_imag.show()
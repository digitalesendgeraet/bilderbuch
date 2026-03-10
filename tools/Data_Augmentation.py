from PIL import Image, ImageEnhance, ImageFilter
import os
import json

def write_json(data, filename="goals.json"):
    # allow passing string OR dict
    if isinstance(data, str):
        data = json.loads(data.replace("'", '"'))

    with open(filename, 'r+') as file:
        file_data = json.load(file)
        file_data["pictures"].update(data)
        file.seek(0)
        json.dump(file_data, file, indent=4)
        file.truncate()



def flip(dir_v, dir_n):
    n = 0
    for images in os.listdir(dir_v):
        img = Image.open(str(dir_v) + "/" + images)

        flip_img = img.transpose(Image.FLIP_LEFT_RIGHT)
        flip_img.save(str(dir_n)+"/" + str(images)[:-4] + "_fliped.png")

        img.save(str(dir_n)+"/" + str(images))

        n += 1

def lighting(dir_v, dir_n):
    n = 0
    for images in os.listdir(dir_v):
        img = Image.open(str(dir_v) + "/" + images)

        enhancer = ImageEnhance.Brightness(img)
        img_bright = enhancer.enhance(1.3)
        img_dark = enhancer.enhance(0.7)
        
        img_bright.save(str(dir_n)+"/" + str(images)[:-4] + "_bright.png")
        img_dark.save(str(dir_n)+"/" + str(images)[:-4] + "_dark.png")

        img.save(str(dir_n)+"/" + str(images))

        n += 1

def rotate(dir_v, dir_n):
    n = 0
    for images in os.listdir(dir_v):
        img = Image.open(str(dir_v) + "/" + images)

        img_rotate_pos = img.rotate(10)
        img_rotate_neg = img.rotate(350)
        
        img_rotate_pos.save(str(dir_n)+"/" + str(images)[:-4] + "_rotPos.png")
        img_rotate_neg.save(str(dir_n)+"/" + str(images)[:-4] + "_rotNeg.png")

        img.save(str(dir_n)+"/" + str(images))

        n += 1


def blurr(dir_v, dir_n):
    n = 0
    for images in os.listdir(dir_v):
        img = Image.open(str(dir_v) + "/" + images)

        img_blurr = img.filter(ImageFilter.BLUR)

        img_blurr.save(str(dir_n)+"/" + str(images)[:-4] + "_blurr.png")

        img.save(str(dir_n)+"/" + str(images))

        n += 1

def contrast(dir_v,dir_n):
        n = 0
        for images in os.listdir(dir_v):
            img = Image.open(str(dir_v) + "/" + images)

            enhancer = ImageEnhance.Contrast(img)
            img_high_contrast = enhancer.enhance(1.3)
            
            img_high_contrast.save(str(dir_n)+"/" + str(images)[:-4] + "_contrast.png")
            img.save(str(dir_n)+"/" + str(images))

            n += 1



def goals(dir):
    with open('goals.json', 'r') as goals:
        data = json.load(goals)
    pictures = data['pictures']
    data = {}

    for images in os.listdir(dir):
        name = str(images)
        n = name.find("_")
        name = name[:n] + name[n+1:]
        if name.find("_") != -1:
            m = name.find("_")
            mainImg = str(images)[:m+1] + ".png"

            goal = pictures[mainImg]["goal"]

            data.update({str(images): {"goal": goal}})

    write_json(data)




# flip("formated_images", "folder1")
# lighting("folder1", "folder2")
# rotate("folder2", "folder3")
# contrast("folder3", "folder4")
goals("folder4")

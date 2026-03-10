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



def deletion():
    with open('goals.json', 'r') as goals:
        data = json.load(goals)

    pictures = data['pictures']
    n=0
    for files in pictures:
        goal = pictures[files]["goal"]
        if goal==-1:
            os.remove("formated_images/" + str(files))
        n+=1
    
# for n in range(500, 549):
#     newData = {"test_" + str(n) + ".png": {"goal": 0}}
#     write_json(newData)

deletion()


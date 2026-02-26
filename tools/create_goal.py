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


file = "goals.json"

print("start")

data = {
    "pictures": {}
}

json_str = json.dumps(data, indent=4)
with open(file, "w") as f:
    f.write(json_str)

for i in range(501):
    file_name = f"test_{i}.png"   # FIX: was a tuple before

    if i % 2 == 0:
        goal = 1
    else:
        goal = 0

    data = {file_name: {"goal": goal}}

    write_json(data)   # FIX: removed str()

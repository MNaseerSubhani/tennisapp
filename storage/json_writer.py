import json


def save(data, path="output.json"):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

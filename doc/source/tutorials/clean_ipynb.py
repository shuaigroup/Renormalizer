import json
import glob

for notebook in glob.glob("*.ipynb"):
    with open(notebook) as f:
        data = json.load(f)
    del data["metadata"]["kernelspec"]
    with open(notebook, "w") as f:
        json.dump(data, f)

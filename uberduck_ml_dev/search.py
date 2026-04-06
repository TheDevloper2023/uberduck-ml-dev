import os

repo_path = r"D:\uberduck-ml-dev"

for root, dirs, files in os.walk(repo_path):
    for file in files:
        if file.endswith(".py"):
            path = os.path.join(root, file)
            with open(path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f, 1):
                    if "librosa" in line:
                        print(f"{path}:{i}: {line.strip()}")
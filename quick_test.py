import os

path = os.path.join(os.getcwd(), 'lfw')

count = 0
for folder in os.listdir(path):
    folder_path = os.path.join(path, folder)
    if len(os.listdir(folder_path))==4: count += 1

print(count)
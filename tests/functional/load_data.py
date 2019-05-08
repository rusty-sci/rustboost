from termcolor import colored, cprint
import os
from urllib.request import urlopen

PATH = "../data/loaded/"

links = {
  "classification": [
    {
      "link": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/iris.scale",
      "name": "iris"
    }
  ]
}

class MSG_TYPE:
  info = colored("[INFO]", "grey", attrs=["bold"])
  error = colored("[ERROR]", "red", attrs=["bold"])

def main():
  load_data()

def create_folders():
  print(MSG_TYPE.info + ": 📁   Creating data folders.")
  if os.path.exists(PATH):
    print(MSG_TYPE.info + ": Folders already exists.")
  else:
    os.makedirs(PATH)
    print(MSG_TYPE.info + ": Folders have been created.")

def load_data():
  for task in links:
    for dataset in links[task]:
      link = dataset["link"]
      name = dataset["name"]
      try:
        print(MSG_TYPE.info + ": 💽   %s data loading." %(name))
        f = urlopen(link)
        data = f.read()
        filename = PATH + name + "_" + task + ".libsvm"
        save_to_libsvm(filename, data)
        print(MSG_TYPE.info + ": ✅   Data saved successfully.")
      except:
        print(MSG_TYPE.error + ": ❌   Error loading %s data" %(name))

def save_to_libsvm(filename, data):
  with open(filename, "w") as f:
    f.write(data.decode('utf-8'))




if __name__ == "__main__":
  create_folders()
  main()
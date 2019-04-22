import sys
from termcolor import colored, cprint
import numpy as np
import argparse
import os
from sklearn.datasets import make_regression

TASK = "regression"
PATH = "../data/generated/"

N_SAMPLES = 30
N_FEATURES = 3
N_INFORMATIVE = 2

class MSG_TYPE:
  info = colored("[INFO]", "grey", attrs=["bold"])
  error = colored("[ERROR]", "red", attrs=["bold"])

def main():
  if TASK == "regression":
    print(MSG_TYPE.info + ": 💽   Generating data for regression. (path: %s)" %(PATH))
    gen_reg()
  elif TASK == "classification":
    pass
  else:
    print(MSG_TYPE.error + ": ❌   No such task. Choose regression or classification.")

def create_folders():
  print(MSG_TYPE.info + ": 📁   Creating data folders.")
  if os.path.exists(PATH):
    print(MSG_TYPE.info + ": Folders already exists.")
  else:
    os.makedirs(PATH)
    print(MSG_TYPE.info + ": Folders have been created.")

def gen_reg():
  data = make_regression(n_samples=N_SAMPLES,
    n_features=N_FEATURES, n_informative=N_INFORMATIVE)
  print(MSG_TYPE.info + ": ✅   Data generated successfully.")
  filename = PATH + TASK + ".libsvm"
  save_to_libsvm(filename, data[0], data[1])
  print(MSG_TYPE.info + ": 📝   Data saved successfully to file: %s" %(filename))

def save_to_libsvm(filename, data, target):
  text_data = ""
  with open(filename, "w") as f:
    for i, sample in enumerate(data):
      text_data += str(target[i])
      for j, feature in enumerate(sample):
        text_data += " " + str(j) + ":" + str(feature)
      text_data += "\n"
    f.write(text_data)




if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Generate artificial data for testing rustboost.")
  parser.add_argument('--task', default=TASK, help='Choose task (default: regression)')
  args = parser.parse_args()
  TASK = args.task
  create_folders()
  main()
import os
import sys

def run_in(device,optimizer):
  print("device")
  if optimizer ="vadam":
    if device == "0":
      commands = open("./commands0.txt").readlines()
      for command in commands:
        print("LAUNCHING", command)
        os.system(command)
    elif device == "1":
      commands = open("./commands1.txt").readlines()
      for command in commands:
        print("LAUNCHING", command)
        os.system(command)
    elif device =="2" :
      commands = open("./commands2.txt").readlines()
      for command in commands:
        print("LAUNCHING", command)
        os.system(command)
    elif device =="3":
      commands = open("./commands3.txt").readlines()
      for command in commands:
        print("LAUNCHING", command)
        os.system(command)
  elif optimizer ="adam":
    if device == "0":
      commands = open("./commands0A.txt").readlines()
      for command in commands:
        print("LAUNCHING", command)
        os.system(command)
    elif device == "1":
      commands = open("./commands1A.txt").readlines()
      for command in commands:
        print("LAUNCHING", command)
        os.system(command)
    elif device =="2" :
      commands = open("./commands2A.txt").readlines()
      for command in commands:
        print("LAUNCHING", command)
        os.system(command)
    elif device =="3":
      commands = open("./commands3A.txt").readlines()
      for command in commands:
        print("LAUNCHING", command)
        os.system(command)


print(sys.argv[1])
run_in(sys.argv[1], sys.argv[2])
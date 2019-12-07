import os
import sys

def run_in(device):
  print("device")

  if device == "0":
    commands = open("./commands0.txt").readlines()
    for command in commands:
      os.system(command)
  elif device == "1":
    commands = open("./commands1.txt").readlines()
    for command in commands:
      os.system(command)
  elif device =="2" :
    commands = open("./commands2.txt").readlines()
    for command in commands:
      os.system(command)
  elif device =="3":
    commands = open("./commands3.txt").readlines()
    for command in commands:
      os.system(command)


print(sys.argv[1])
run_in(sys.argv[1])
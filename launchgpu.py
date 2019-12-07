import mainmp
import sys

def run_in(device):
  print("device")
  if device == "0":
    num_k = [4,9,14,19]
    num_kq = [4,9,14,19,24,29,34]
    device = 0
    for nk in num_k:
      for nkq in num_kq:
        mainmp.mainmp(nk,nkq,1)
  if device == "1":
    num_k = [4,9,14, 19]
    num_kq = [4,9,14,19,24,29,34]
    device = 1
    for nk in num_k:
      for nkq in num_kq:
        mainmp.mainmp(nk,nkq,1)
  elif device =="2" :
    num_k = [24,29]
    num_kq = [4,9,14,19,24,29,34]
    device = 2
    for nk in num_k:
      for nkq in num_kq:
        mainmp.mainmp(nk,nkq,2)
  elif device =="3":
    num_k = [34]
    num_kq = [4,9,14,19,24,29,34]
    device = 3
    for nk in num_k:
      for nkq in num_kq:
        mainmp.mainmp(nk,nkq,3)

print(sys.argv[1])
run_in(sys.argv[1])
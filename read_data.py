import pickle
import pprint
import json

file = open("./dataset/AIR/1-1.pkl","rb")
jfile = open("./a,json",'w')
data = pickle.load(file)
json.dump(data,jfile,ensure_ascii=False,sort_keys=True,indent=4)
# pprint.pprint(data)
file.close()
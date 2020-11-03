import os
import wget
import pandas as pd


root_dir = "/work/bjuncek/AVA_actions/"
trainval_url = "https://s3.amazonaws.com/ava-dataset/annotations/ava_file_names_trainval_v2.1.txt"
test_url = "https://s3.amazonaws.com/ava-dataset/annotations/ava_file_names_test_v2.1.txt"


trainval_path = os.path.join(root_dir, "ava_file_names_trainval_v2.1.txt")
test_path = os.path.join(root_dir, "ava_file_names_test_v2.1.txt")

if not os.path.exists(trainval_path):
    wget.download(trainval_url, trainval_path)
    
if not os.path.exists(test_path):
    wget.download(test_url, test_path)
    
data_dir = os.path.join(root_dir, "data")
os.makedirs(data_dir, exist_ok=True)
for split in ["train", "val", "test"]:
    data_dl = os.path.join(data_dir, split)
    os.makedirs(data_dl, exist_ok=True)
    path = os.path.join(root_dir,f"ava_{split}_v2.2.csv")
    df = pd.read_csv(path, names=["video_id", "middle_frame_timestamp", "x1", "y1", "x2", "y2", "action_id", "person_id"])
    if split in ['train', 'val']:
        files = pd.read_csv(trainval_path, names=['path'])
        url_template = "https://s3.amazonaws.com/ava-dataset/trainval/{file}"
    
    for file in files.path:
        if file.split(".")[0] in df.video_id.unique():
            out_path= os.path.join(data_dl, file)
            if not os.path.exists(out_path):
                try:
                    wget.download(url=url_template.format(file=file), out=out_path, bar=wget.bar_thermometer)
                except:
                    continue
        else:
            continue
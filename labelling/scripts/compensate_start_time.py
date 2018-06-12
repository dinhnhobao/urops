import os
import datetime
import argparse
import pandas as pd


ap = argparse.ArgumentParser()
ap.add_argument("-d", "--date", required=True, help="Date of pictures")
args = vars(ap.parse_args())
date = args["date"]

all_start_times = pd.read_csv("/Users/nurmister/Documents/academic/urops/labelling/scripts/start_times.csv")
actual_start_time = all_start_times[all_start_times["date"] == date]["time"].to_string()
actual_start_time = actual_start_time[(len(actual_start_time) - 4):len(actual_start_time)]

def convert_pict_to_time_delta(pict):
    time = pict[len(pict)-8:len(pict)-4]
    return (time, datetime.datetime.strptime(time, "%H%M") - 
            datetime.datetime.strptime(actual_start_time, "%H%M"))
    
times = list()
for pict in os.listdir("../pictures_dump/"):
    if pict != ".DS_Store":
        times.append(convert_pict_to_time_delta(pict))
compensation = sorted(times, key=lambda x: x[1])[0][1]

def apply_compensation(pict):
    time = pict[len(pict)-8:len(pict)-4]
    compensated_delta = str(datetime.datetime.strptime(time, "%H%M") - compensation)
    new_time = compensated_delta[len(compensated_delta)-8:len(compensated_delta)-6] + compensated_delta[len(compensated_delta)-5:len(compensated_delta)-3] 
    return pict[0:len(pict)-8] + new_time + ".jpg"

for pict in os.listdir("../pictures_dump/"):
    if pict != ".DS_Store":
        os.rename(os.path.join("../pictures_dump/", pict), os.path.join("../pictures_dump/", apply_compensation(pict)))
import os

import pandas as pd


os.mkdir("../pictures_dump/cropped")

instructions = pd.read_csv("dummy_crop_instructions.csv", index_col=False)

for pict in os.listdir("../pictures_dump/"):
    for spot_number in range(54):
        os.system("python crop.py --image ../pictures_dump/" + pict
                  + " --angle " + str(instructions["angle"][spot_number])
                  + " --x_one " + str(instructions["x_one"][spot_number])
                  + " --x_two " + str(instructions["x_two"][spot_number])
                  + " --y_one " + str(instructions["y_one"][spot_number])
                  + " --y_two " + str(instructions["y_two"][spot_number])
                  + " --label " + str(instructions["label"][spot_number]))

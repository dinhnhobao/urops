import os

import pandas as pd

instructions = pd.read_csv("dummy_crop_instructions.csv", )

os.mkdir("PICTURES/cropped")
for spot_number in range(55):
    os.system("python crop.py --image " + + " --angle " + + " --x_one " + + " --x_two " + + " --y_one " + + " --y_two " + + " --label " + )

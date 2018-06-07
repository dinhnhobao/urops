import os

import pandas as pd


def main():

    os.mkdir("../pictures_dump/cropped")

    instructions = pd.read_csv("dummy_crop_instructions.csv", index_col=False)

    num_done = 0
    pictures = os.listdir("../pictures_dump/")
    for image in pictures:
        if image != ".DS_Store":
            num_done += 1
            for spot_number in range(54):
                print("python crop.py --image " + image
                      + " --angle " + str(instructions["angle"][spot_number])
                      + " --x_one " + str(instructions["x_one"][spot_number])
                      + " --x_two " + str(instructions["x_two"][spot_number])
                      + " --y_one " + str(instructions["y_one"][spot_number])
                      + " --y_two " + str(instructions["y_two"][spot_number])
                      + " --label " + str(instructions["label"][spot_number]))
            print("echo " + str(num_done) + "/" + str(len(pictures)))


if __name__ == "__main__":
    main()

import os

import pandas as pd

image_type = '.jpg'  # Set before execution.
num_spots = 36  # Number of spots in each full image to crop.


def main():
    """Generates shell commands to obtain cropped images from each full image
    in '../pictures_dump'."""
    # Fill 'crop_instructions.csv' with the help of
    # "get_spot_coords_and_angles.py".
    instructions = pd.read_csv('crop_instructions.csv', index_col=False)
    os.mkdir('if [ ! -d "../pictures_dump/cropped" ]; then mkdir data; fi;')

    num_done = 0
    all_pictures = os.listdir('../pictures_dump/')
    for picture in all_pictures:
        if picture.endswith(image_type):
            num_done += 1
            for spot_number in range(num_spots):
                print('python crop.py --image ' + picture
                      + ' --angle ' + str(instructions['angle'][spot_number])
                      + ' --x_one ' + str(instructions['x_one'][spot_number])
                      + ' --x_two ' + str(instructions['x_two'][spot_number])
                      + ' --y_one ' + str(instructions['y_one'][spot_number])
                      + ' --y_two ' + str(instructions['y_two'][spot_number])
                      + ' --label ' + str(instructions['label'][spot_number]))
            print('echo ' + str(num_done) + '/' + str(len(all_pictures)))


if __name__ == '__main__':
    main()

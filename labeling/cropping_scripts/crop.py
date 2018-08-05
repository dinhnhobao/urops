import argparse
import math
import cv2


def rotate_image(image, angle):
    image_height = image.shape[0]
    image_width = image.shape[1]
    diagonal_square = (image_width*image_width) + (image_height*image_height)

    diagonal = round(math.sqrt(diagonal_square))
    padding_top = round((diagonal-image_height) / 2)
    padding_bottom = round((diagonal-image_height) / 2)
    padding_right = round((diagonal-image_width) / 2)
    padding_left = round((diagonal-image_width) / 2)
    padded_image = cv2.copyMakeBorder(image,
                                      top=padding_top,
                                      bottom=padding_bottom,
                                      left=padding_left,
                                      right=padding_right,
                                      borderType=cv2.BORDER_CONSTANT,
                                      value=0)
    padded_height = padded_image.shape[0]
    padded_width = padded_image.shape[1]
    transform_matrix = cv2.getRotationMatrix2D(
                (padded_height/2,
                 padded_width/2),
                angle,
                1.0)
    rotated_image = cv2.warpAffine(padded_image,
                                   transform_matrix,
                                   (diagonal, diagonal),
                                   flags=cv2.INTER_LANCZOS4)
    return rotated_image


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
ap.add_argument("-a", "--angle", required=True, help="Angle to rotate before cropping")
ap.add_argument("--x_one", required=True, help="Top-right x-coordinate")
ap.add_argument("--y_one", required=True, help="Top-right y-coordinate")
ap.add_argument("--x_two", required=True, help="Bottom-left x-coordinate")
ap.add_argument("--y_two", required=True, help="Bottom-left y-coordinate")
ap.add_argument("-l", "--label", required=True, help="Spot number")

args = vars(ap.parse_args())

# Crops image of spot.
cropped_image = rotate_image(cv2.imread("../pictures_dump/" + args["image"]),
                             int(args["angle"]))[
                             int(args["y_one"]):int(args["y_two"]),
                             int(args["x_one"]):int(args["x_two"])]

# Resizes the cropped image to 128 by 128.
resized_image = cv2.resize(cropped_image, (128, 128))

try:
    cv2.imwrite("../pictures_dump/cropped/"
                + args["image"][0:(len(args["image"]) - 4)]
                + "_" + args["label"] + ".jpg", resized_image)
except:
    print("Failed.")

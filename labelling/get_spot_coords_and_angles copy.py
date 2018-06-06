import argparse
import cv2
import math

# Initialize the list of reference points and boolean indicating
# whether cropping is being performed or not.
refPt = []
cropping = False

def click_and_crop(event, x, y, flags, param):
    global refPt, cropping

    # If the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed.
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        cropping = True

    # Check to see if the left mouse button was released.
    elif event == cv2.EVENT_LBUTTONUP:
        # Record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished.
        refPt.append((x, y))
        cropping = False

        # Draw a rectangle around the region of interest.
        cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)

        print("[y:y+h, x:x+w]: [%d:%d, %d:%d]" %
              (refPt[0][1],
              refPt[1][1],
              refPt[0][0],
              refPt[1][0]))
        cv2.imshow("image", image)


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
    print("Angle: %d" % angle)
    return rotated_image


# Construct the argument parser and parse the arguments.
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

# Load the image, clone it, and setup the mouse callback function.
image = rotate_image(cv2.imread(args["image"]), 51)
clone = image.copy()
cv2.namedWindow("image", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("image", click_and_crop)


# Keep looping until the 'q' key is pressed.
while True:
    # Display the image and wait for a keypress.
    cv2.imshow("image", image)
    key = cv2.waitKey(1) & 0xFF

    # If the 'r' key is pressed, reset the cropping region.
    if key == ord("r"):
        image = clone.copy()

    # If the 'c' key is pressed, break from the loop.
    elif key == ord("c"):
        break

# If there are two reference points, then crop the region of interest
# from the image and display it.
if len(refPt) == 2:
    roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
    cv2.imshow("ROI", roi)
    cv2.waitKey(0)

cv2.destroyAllWindows()

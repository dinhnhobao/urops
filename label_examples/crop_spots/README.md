There are two key steps behind using these scripts:

1. Set `crop_instructions.csv` by applying `get_spot_coords_and_angles.py`
to a template full image, which is in this case `cropping_spot_IDs.png`.
This should be a one-time operation, given all pictures to be cropped are
taken from the same perspective.
2. Place all images to be cropped in `../pictures_dump/`, and then run:
`$ source ./dailycropper.sh` to crop each full image; results will be
stored in `../pictures_dump/cropped/`.

To control finer aspects of cropping, like cropped image size, edit
`crop.py` and `crop_all_spots.py`; the other scripts will likely not
need modification.

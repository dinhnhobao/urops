After obtaining the cropped images using the scripts of `../crop_spots/`,
use the scripts here to label them.

1. If the directories do not already exist, create `pictures_to_label/` and
`label_csvs/`. Transfer the cropped images from `../pictures_dump/cropped/`
to `pictures_to_label/`.
2. Use `spot_labeller.R` to create a labels `.csv` file in `labels_csv/`.
3. Store the labels `.csv` file in `../../data/labels/` and the cropped images
in `../../data/cropped/cropped_{date}/`, where date is the date these pictures
were taken.
4. The data is now ready for serialization using
`../../save_date/serialize_dataset.ipynb` if you do not wish to add any more
day's worth of data to `../../data/`.

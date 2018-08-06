This is the public Google Drive folder for the NUSLot dataset.

The NUSLot dataset was created in the summer of 2018 for an undergraduate
research project into the application of convolutional neural networks to
ascertain the occupancy of a parking lot by classifying its spots as either
"occupied" or "empty".

The data in this folder has already been pre-processed to that end; read the
documentation at github.com/nurmister/urops to understand how to load and use
this dataset for machine-learning purposes. Briefly,
1. "raw_data" comprises of full parking lot images, individual spot-crops, and
corresponding spot labels ("empty" -> 0, "occupied" -> 1).
2. "full_dataset" comprises of serialized NumPy arrays representing
training, validation, and testing sets good for stratified three-fold
cross-validation. There are in total 50,000 unique examples for training and
evaluation.
3. "toy_dataset" is a 10% simple random sample of the data of
"full_dataset"; i.e. it contains 5,000 unique examples.

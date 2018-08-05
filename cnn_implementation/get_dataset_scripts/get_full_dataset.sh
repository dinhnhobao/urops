mkdir data;
chmod +x get_dataset_scripts/gdown.pl;
./get_dataset_scripts/gdown.pl https://drive.google.com/file/d/1iQjTh7pRVrBkqB4VlwRJEyTxwOG2QfFQ/view?usp=sharing data/full_dataset/test_set.hdf5;
./get_dataset_scripts/gdown.pl https://drive.google.com/file/d/1E-Fxg5HLFUuxqNkbQh4kC9lsZPMdg3NL/view?usp=sharing data/full_dataset/train_validation_set_1.hdf5;
./get_dataset_scripts/gdown.pl https://drive.google.com/file/d/1emp9WVFJ5ctogHNIumKglILgTMqb22O3/view?usp=sharing data/full_dataset/train_validation_set_2.hdf5;
./get_dataset_scripts/gdown.pl https://drive.google.com/file/d/1BfQYvDrRbttm2IUAgmOfYr6h3oMOqTv8/view?usp=sharing data/full_dataset/train_validation_set_3.hdf5;
mkdir data;
chmod +x get_dataset_scripts/gdown.pl;
./get_dataset_scripts/gdown.pl https://drive.google.com/file/d/1ykdpyZpps3gUccRlXzKh0romGBYNSlPO/view?usp=sharing data/toy_dataset/test_set.hdf5;
./get_dataset_scripts/gdown.pl https://drive.google.com/file/d/1pGn2tdd4RAYL4YsDkSuIQEaPDCuZhA3q/view?usp=sharing data/toy_dataset/train_validation_set_1.hdf5;
./get_dataset_scripts/gdown.pl https://drive.google.com/file/d/1I-2B-41PYKTzJeNm2blalUMDzA7Hmtyx/view?usp=sharing data/toy_dataset/train_validation_set_2.hdf5;
./get_dataset_scripts/gdown.pl https://drive.google.com/file/d/18DbqjS0xjoMyrAunqgC2CKrI5EyPkHnQ/view?usp=sharing data/toy_dataset/train_validation_set_3.hdf5;
mkdir data;
chmod +x get_dataset_scripts/gdown.pl;
./get_dataset_scripts/gdown.pl https://drive.google.com/file/d/1WdtrMytKh4S3PEo9O51Y0Wv3Cko0Is6S/view?usp=sharing data/test_set.hdf5;
./get_dataset_scripts/gdown.pl https://drive.google.com/file/d/1nqLlcKXqDtlpeQhksMRsetSrQrI7we3e/view?usp=sharing data/toy_dataset/train_validation_set_1.hdf5;
./get_dataset_scripts/gdown.pl https://drive.google.com/file/d/1V_XxHkUY0dTlkBpFpSw0VhPPNTYU9Q-L/view?usp=sharing data/toy_dataset/train_validation_set_2.hdf5;
./get_dataset_scripts/gdown.pl https://drive.google.com/file/d/1KR0jotLHhUSLxeVOxEkf_VdMRD2UslAu/view?usp=sharing data/toy_dataset/train_validation_set_3.hdf5;
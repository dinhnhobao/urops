Here lies the CNN detailed in `../writeup/main.tex`.

Prior to execution, if no data is present, you may download
it by calling the script:

`$ python cnn.py --download_data True --dataset_name $DESIRED_DATASET`

where `$DESIRED_DATASET` is a dataset downloadable in the manner of\*
`toy_dataset` and `full_dataset`.

\*I.e. has a shell script in the form of `get_$DESIRED_DATASET.sh` in
`get_dataset_scripts/` that downloads data to `../data/$DESIRED_DATASET`
in the form created by `../save_dataset/serialize_dataset.ipynb`.

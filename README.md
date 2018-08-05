### Repository for the AY 17/18 UROPS on ascertaining the occupancy of a parking lot using convolutional neural networks.
*Clone with `--depth 1` to avoid a large `.git` folder; required datasets can be found [here](https://drive.google.com/open?id=1_UIeomO1Pn8GsVkRkuybTSo_R3cvOh_-).*

<p align="center"> 
<img src="https://lh3.googleusercontent.com/AnfeQJq_ZAzAAp1keX0NAS_6vOby5PoXql_iZ0Udg91m0Y0kH4Grwzrhh_LdZcK0w7NcXOpyFEoMhLgehMHj=w2880-h1800">
</p>

Refer to `writeup/main.tex` for the context behind and the usage of this repository.

Nonetheless, a brief overview:

1. `NUSLot_google_drive_details/` points to the *NUSLot* dataset, which comprises of two
datasets: `toy_dataset` and `full_dataset`.
2. `cnn_implementation/` contains the convolutional neural network Tensorflow implementation
trained and tested on the *NUSLot* dataset.
3. `label_examples/` contains scripts to crop example-images from full images, and assign labels
to them.
4. `pi/` contains configuration details of the Raspberry Pi used to obtain the data behind *NUSLot*. 
5. `save_dataset/` contains a script to serialize labeled image-based data, and was used to
create *NUSLot/{full, toy}_dataset*

Feel free to raise issues.

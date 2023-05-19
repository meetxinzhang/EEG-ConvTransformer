# EEG-ConvTransformer

This is a community implemention of arcticle:

`[1] Bagchi S, Bathula D R. EEG-ConvTransformer for single-trial EEG-based visual stimulus classification[J]. Pattern Recognition, 2022, 129: 108757.`

The proposed method (Called EEG-ConvTransformer) of citation[1] is implemented in /model. It should be no problem.

## Usage
Before training, you need:

1) Download the dataset from https://purl.stanford.edu/bq914sc3730
2) Download the electrodes XYZ locations from: ftp://ftp.egi.com/pub/support/Documents/net_layouts/hcgsn_128.pdf 
Note that I can not download this PDF, fortunately, @yellow006 send me a copy, thanks. 
There is also a 64-channels version from https://github.com/numediart/EEGLearn-Pytorch/blob/master/Sample%20Data/Neuroscan_locs_orig.mat
It depends on what dataset you use.
3) Perform pre-processing by run data_load/serialize.py. This script will do `Azimuthal Equidistant Projection(AEP)[2]` and pkl serialization in Parallel. 
The EEG raw data will be transformed to images[1][2].
Remember to pass two filepath.

Now, You can run main.py to start your training task. The testing or Validation can be easily written by modifying this file.

### Acknowledgement
Thanks to @yellow006 for the GSN-HydroCel-128.xlsx file.

### Refs

[1] Bagchi S, Bathula D R. EEG-ConvTransformer for single-trial EEG-based visual stimulus classification[J]. Pattern Recognition, 2022, 129: 108757.

[2] Bashivan, et al. "Learning Representations from EEG with Deep Recurrent-Convolutional Neural Networks." International conference on learning representations (2016).

[3] https://github.com/pbashivan/EEGLearn

[4] https://github.com/numediart/EEGLearn-Pytorch

## Note
Any question and issue is welcome!

This project is in the collection: https://github.com/szu-advtech/AdvTech/tree/main/2022/3-%E5%BC%A0%E6%AC%A3%20%E6%8C%87%E5%AF%BC%E8%80%81%E5%B8%88-%E9%92%9F%E5%9C%A3%E5%8D%8E-EEG-ConvTransformer


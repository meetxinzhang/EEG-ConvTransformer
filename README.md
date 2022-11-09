# EEG-ConvTransformer

This is a community implemention of arcticle:

`[1] Bagchi S, Bathula D R. EEG-ConvTransformer for single-trial EEG-based visual stimulus classification[J]. Pattern Recognition, 2022, 129: 108757.`

The proposed method (Called EEG-ConvTransformer) of citation[1] is implemented in /model. It should be no problem.

Before running, the visualized-image should be generated from EEG signals by run /preprocess/project2img.ipynb, read chapter 3.1 of citation[1] for more details.
Note that in this part there are some uncertain coding due to undisclosed details in citation [2]. It's welcome to help me to refine this repository.

I referenced citation[2] for the implemention of Azimuthal Equidistant Projection:

`[2] Bashivan, et al. "Learning Representations from EEG with Deep Recurrent-Convolutional Neural Networks." International conference on learning representations (2016).`


And the following repositories are also referenced:

https://github.com/pbashivan/EEGLearn

https://github.com/numediart/EEGLearn-Pytorch


This is a uncompleted implementation, welcome to commit your request and issues.

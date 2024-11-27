# IPCV_Colorize_BW_Images

## Introduction

This project was developed as part of the "Image Processing and Computer Vision" course at Politecnico di Torino. I chose the topic of black-and-white image colorization to combine my knowledge of OpenCV with a personal goal: colorizing old family photos. 

The goal of this study is to use a pre-trained network and compare its results with Photoshop up to November 2024. Additionally, the study explores how various image processing techniques, such as denoising, histogram equalization, and grain removal, affect the colorization process.

This project also provided an opportunity to explore neural networks for the first time. While I am aware that more advanced models exist today, I opted to base my work on the 2016 study ["Colorful Image Colorization"](http://richzhang.github.io/colorization/) for simplicity and to gain a better understanding of foundational approaches in this field.

## Inspiration

This project was inspired by the article ["Black and White Image Colorization with OpenCV and Deep Learning"](https://pyimagesearch.com/2019/02/25/black-and-white-image-colorization-with-opencv-and-deep-learning/) published on PyImageSearch. 

The code in this repository builds upon the concepts and techniques described in the article, with additional modifications and enhancements to suit specific project goals. Some sections of the code were adapted or derived from examples provided in the article. Please refer to the original article for further details.

## Scripts Overview

1. **bw2color_single_image**: This script allows users to colorize a single image. It is ideal for those who want to evaluate the effect of the colorization process on a specific image. Simply provide the image path, and the script will output the result.

2. **bw2color_image_folder**: If you want to colorize multiple images and save the output properly, use the **Image Folder Script**. This script processes all images in a folder and saves the results accordingly.

3. **benchmark_generator**: This script allows users to run multiple tests and compare the results in one go. It is useful for benchmarking the effect of different techniques (such as denoising or histogram equalization) on the colorization process. It will automatically generate a comparison of results for easy evaluation.

## Model Source

The pre-trained Caffe model used in this project was downloaded from the description of the video ["How to Colorize Black and White Photos in Python"](https://www.youtube.com/watch?v=gAmskBNz_Vc). 

Special thanks to the video creator for providing the model and code.

## Acknowledgements

This project is derived from:
- [Colorful Image Colorization](http://richzhang.github.io/colorization/), licensed under BSD 2-Clause.

The original copyright belongs to [Original Authors](https://github.com/richzhang/colorization/commits?author=richzhang). The modifications in this repository are licensed under [The 2-Clause BSD License](https://opensource.org/license/bsd-2-clause).

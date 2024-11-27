import numpy as np
import argparse
import cv2
import matplotlib.pyplot as plt
import input_preprocess

# This script processes a single black-and-white image at a time, applying colorization using a pre-trained neural
# network model. The script allows additional preprocessing steps such as histogram equalization, denoising,
# and grain/scratches removal, depending on the flags provided. It doesn't save the output but instead shows the
# original and colorized images side by side. The input image must be specified as a path along with the model files.
# The output will be displayed using matplotlib's pyplot.

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, required=True,
                help="path to input black and white image")
ap.add_argument("-p", "--prototxt", type=str, required=True,
                help="path to Caffe prototxt file")
ap.add_argument("-m", "--model", type=str, required=True,
                help="path to Caffe pre-trained model")
ap.add_argument("-c", "--points", type=str, required=True,
                help="path to cluster center points")
ap.add_argument("--equalizeHist", action="store_true",
                help="apply histogram equalization to the input image before coloring")
ap.add_argument("--denoise", action="store_true",
                help="denoise the image before coloring")
ap.add_argument("--removeGrainAndScratches", action="store_true",
                help="remove grain and scratches")
args = vars(ap.parse_args())

# load the model and cluster center points
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
pts = np.load(args["points"])

# add the cluster centers as 1x1 convolutions to the model
class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
pts = pts.transpose().reshape(2, 313, 1, 1)
net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

# load the input image from disk, scale the pixel intensities to the range [0, 1], and then convert the image from
# the BGR to Lab color space
image = cv2.imread(args["image"])

# save the original image before any pre colorization operation
original_image = image

# Apply histogram equalization if the flag is set
if args["equalizeHist"]:
    print("[INFO] Applying histogram equalization...")
    original_image = image
    image = input_preprocess.equalize_bgr_image(image)

# Apply denoising if the flag is set
if args["denoise"]:
    print("[INFO] Applying denoising...")
    original_image = image
    image = input_preprocess.simple_denoise(image)

# Remove grain and scratches if the flag is set
if args["removeGrainAndScratches"]:
    print("[INFO] Removing grain and stretches...")
    original_image = image
    image = input_preprocess.remove_grain_and_scratches(image)

# Scale the pixel intensities and convert to Lab color space
scaled = image.astype("float32") / 255.0
lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

# resize the Lab image to 224x224 (the dimensions the colorization network accepts), split channels, extract the 'L'
# channel, and then perform mean centering
resized = cv2.resize(lab, (224, 224))
L = cv2.split(resized)[0]
L -= 50

# pass the L channel through the network which will *predict* the 'a' and 'b' channel values
net.setInput(cv2.dnn.blobFromImage(L))
ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
# resize the predicted 'ab' volume to the same dimensions as our input image
ab = cv2.resize(ab, (image.shape[1], image.shape[0]))

# grab the 'L' channel from the *original* input image (not the resized one) and concatenate the original 'L' channel
# with the predicted 'ab' channels
L = cv2.split(lab)[0]
colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)

# convert the output image from the Lab color space to RGB, then clip any values that fall outside the range [0, 1]
colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2RGB)
colorized = np.clip(colorized, 0, 1)

# the current colorized image is represented as a floating point data type in the range [0, 1] -- let's convert to an
# unsigned 8-bit integer representation in the range [0, 255]
colorized = (255 * colorized).astype("uint8")

# convert the input image from the BGR color space to RGB for a correct output during the print
original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

# show the original and output colorized images
plt.subplot(1, 2, 1).axis('off')
plt.imshow(original_image)
plt.title("Original")

plt.subplot(1, 2, 2).axis('off')
plt.imshow(colorized)
plt.title("Colorized")

plt.show()

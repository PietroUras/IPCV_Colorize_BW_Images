import numpy as np
import argparse
import cv2
import os
import input_preprocess
import save_images as save
from PIL import Image

# Given an input folder containing black and white images, an output folder to save the colorized images,
# and several flags expressed through argparse (such as histogram equalization, denoising, and grain removal),
# this script processes each image in the input folder by applying the selected modifications and colorizes the images.
# The processed images are then saved to the specified output folder.

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, required=True,
                help="path to the folder containing black and white images")
ap.add_argument("-o", "--output", type=str, required=True,
                help="path to the output folder to save colorized images")
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
                help="remove grain and scratches before coloring")
args = vars(ap.parse_args())

# Load the model and cluster center points
print("[INFO] Loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
pts = np.load(args["points"])

# Add the cluster centers as 1x1 convolutions to the model
class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
pts = pts.transpose().reshape(2, 313, 1, 1)
net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

# Check if the output directory exists
if not os.path.exists(args["output"]):
    # If not, create the directory
    os.makedirs(args["output"])

# The 'colorized' prefix is used to name the saved file.
output_prefix = "colorized"

# Process all images in the input directory
for filename in os.listdir(args["input"]):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')):
        input_path = os.path.join(args["input"], filename)
        print(f"[INFO] Processing {filename}...")
        image = cv2.imread(input_path)

        # Apply histogram equalization if the flag is set
        if args["equalizeHist"]:
            output_prefix = "equalized_hist"
            print("[INFO] Applying histogram equalization...")
            original_image = image
            image = input_preprocess.equalize_bgr_image(image)
            save.save_input_preprocess(original_image, image, filename, args["output"], "Histogram equalization")

        # Apply denoising if the flag is set
        if args["denoise"]:
            output_prefix = "denoised"
            print("[INFO] Applying denoising...")
            original_image = image
            image = input_preprocess.simple_denoise(image)
            save.save_input_preprocess(original_image, image, filename, args["output"], "Denoising")

        # Remove grain and scratches if the flag is set
        if args["removeGrainAndScratches"]:
            output_prefix = "removedGrainAndScratches_"
            print("[INFO] Removing grain and stretches...")
            original_image = image
            image = input_preprocess.remove_grain_and_scratches(image)
            save.save_input_preprocess(original_image, image, filename, args["output"],
                                        "Denoising & morphological operations")

        # Scale the pixel intensities and convert to Lab color space
        scaled = image.astype("float32") / 255.0
        lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

        # Resize the Lab image, extract L channel, and mean center
        resized = cv2.resize(lab, (224, 224))
        L = cv2.split(resized)[0]
        L -= 50

        # Predict the 'a' and 'b' channels
        net.setInput(cv2.dnn.blobFromImage(L))
        ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
        ab = cv2.resize(ab, (image.shape[1], image.shape[0]))

        # Combine L channel with predicted 'ab' channels
        L = cv2.split(lab)[0]
        colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
        colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2RGB)
        colorized = np.clip(colorized, 0, 1)
        colorized = (255 * colorized).astype("uint8")

        # Save the colorized image
        output_path = os.path.join(args["output"], f"{output_prefix}_{filename}")

        # Handle file extension issues by replacing `.tif` with `.tiff` if necessary
        if output_path.lower().endswith('.tif'):
            output_path = output_path.replace(".tif", ".tiff")

        # Save using Pillow to avoid the 'KeyError' issue with TIF files
        colorized_image = Image.fromarray(colorized)
        colorized_image.save(output_path, format='TIFF')

        print(f"[INFO] Saved colorized image to {output_path}")

print("[INFO] All images processed.")

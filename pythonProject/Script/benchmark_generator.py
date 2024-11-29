import os
import save_images

# Before running this script is necessary to run bw2color_image_folder and update
# properly input_folders and output_path with the correct paths

# folder of the images to display and compare
input_folders = [
    r"E:\PyCharm\Colorization\pythonProject\Images\Input\Full_quality_png",
    r"E:\PyCharm\Colorization\pythonProject\Images\Output\Colorized",
    r"E:\PyCharm\Colorization\pythonProject\Images\Output\Denoise",
    r"E:\PyCharm\Colorization\pythonProject\Images\Output\Hist_EQ",
    r"E:\PyCharm\Colorization\pythonProject\Images\Output\Remove_grain_and_scratches",
    r"E:\PyCharm\Colorization\pythonProject\Images\Output\Photoshop"
]

# folder where to save the benchmark image
output_path = r"E:\PyCharm\Colorization\pythonProject\Images\Output\Benchmark"
os.makedirs(output_path, exist_ok=True)

# Read images from all the folders
images_by_position = save_images.read_images_from_folders(input_folders)

# Concatenate images for each position
save_images.concatenate_images(images_by_position, output_path)

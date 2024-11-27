import cv2
import numpy as np
import os


# All aesthetic parameters such as padding size and font size are based on images with approximate dimensions of
# 3000x4000 pixels. The padding height, font scale, and text thickness were chosen to ensure that the comparison
# images are clear and readable, even for images with large dimensions. Adjusting these parameters might be necessary
# for different image sizes to maintain consistency and legibility.

def save_input_elaboration(original, edited, filename, output, elaboration):
    """
    Saves an image that contains both original and edited inputs, side by side, with text annotations.
    This is useful for comparing the effects of different image processing techniques.

    :param original: image without modification
    :param edited: image with modification
    :param filename: name of the file
    :param output: folder to save the comparison
    :param elaboration: name of the modification
    """

    # Ensure the "Comparison" subfolder exists within the output folder
    comparison_folder = os.path.join(output, "Comparison")
    os.makedirs(comparison_folder, exist_ok=True)  # Create the folder if it doesn't exist

    # Define the font and size for the text
    font = cv2.FONT_HERSHEY_COMPLEX
    font_scale = 5  # Adjust font size for visibility
    color = (0, 0, 0)  # Black color for text
    thickness = 10

    # Calculate the padding size (white space) above the image
    padding_height = 400  # You can adjust this value to get more or less padding
    original_height, original_width = original.shape[:2]
    edited_height, edited_width = edited.shape[:2]

    # Add white padding above the images
    original_padded = np.ones((original_height + padding_height, original_width, 3),
                              dtype=np.uint8) * 255  # White padding
    edited_padded = np.ones((edited_height + padding_height, edited_width, 3), dtype=np.uint8) * 255  # White padding

    # Place the original and edited images below the padding
    original_padded[padding_height:, :] = original
    edited_padded[padding_height:, :] = edited

    # Add text above the images
    cv2.putText(original_padded, f" No {elaboration}", (10, padding_height // 2), font, font_scale, color, thickness)
    cv2.putText(edited_padded, f" With {elaboration}", (10, padding_height // 2), font, font_scale, color, thickness)

    # Add a space between the images by creating an empty space (e.g., a black bar)
    space = np.zeros((max(original_padded.shape[0], edited_padded.shape[0]), 100, 3),
                     dtype=np.uint8)  # Adjust width (50) for spacing

    # Combine original, space, and edited images
    combined_image = np.hstack((original_padded, space, edited_padded))

    # Save the combined image
    output_path = os.path.join(comparison_folder, f"comparison_{elaboration}_{filename}")
    cv2.imwrite(output_path, combined_image)


def read_images_from_folders(folders):
    """
    Reads all images from multiple folders. This function assumes that each folder contains images that correspond
    to the same sequence (e.g., different stages of processing). It collects the images in a way that groups
    images by their corresponding position across all folders.

    :param folders: list of folders containing the images.
    :return: list of lists of images, where each list contains the images from the same position in all folders.
    """
    images_by_position = []  # A list that will hold the images for each position (first, second, etc.)

    # For each folder in the list
    for folder in folders:
        images = []
        for item in os.listdir(folder):
            item_path = os.path.join(folder, item)
            if os.path.isfile(item_path):  # Check if it is a file
                image = cv2.imread(item_path)  # Read the image
                images.append(image)
        images_by_position.append(images)

    return images_by_position


def concatenate_images(images_by_position, output_path):
    """
    Concatenates images from different folders with captions and padding, and arranges them into rows for comparison.
    This function is used to create a side-by-side comparison of different image processing stages (e.g., input vs. colorized,
    denoised, etc.).

    :param images_by_position: list of images for each position.
    :param output_path: path to save the concatenated images.
    """
    num_images = min(len(folder_images) for folder_images in images_by_position)  # Min images across folders
    padding = 20  # Padding between images

    # The captions below are provided as examples, assuming the folders given as input contain images
    # corresponding to different stages of processing. These captions are customizable to match the
    # specific folder names or processing steps you are using in your project.

    caption = [
        "Input",
        "Colorized",
        "Denoised + Colorized",
        "Hist. Equalized + Colorized",
        "Morph. Ops + Colorized",
        "Photoshop Colorization"
    ]

    # Define the font and size for the text
    font = cv2.FONT_HERSHEY_COMPLEX
    font_scale = 5  # Adjust font size for visibility
    text_color = (0, 0, 0)  # Black text
    text_thickness = 10
    caption_padding = 400  # Space for the caption above each image

    for i in range(num_images):
        # Get all images for the i-th position
        images_at_position = [folder_images[i] for folder_images in images_by_position]

        # Ensure all images are resized to the same height
        height = max(img.shape[0] for img in images_at_position)
        resized_images = [
            cv2.resize(img, (int(img.shape[1] * height / img.shape[0]), height))
            for img in images_at_position
        ]

        # Add captions and padding to each image
        def add_caption_and_padding(image, caption_text):
            """
            Adds a caption and top padding to an image.
            """
            # Calculate the width of the caption text
            text_size = cv2.getTextSize(caption_text, font, font_scale, text_thickness)[0]
            text_x = (image.shape[1] - text_size[0]) // 2  # Center the text horizontally
            text_y = caption_padding // 2 + text_size[1]  # Center the text vertically in padding

            # Create a white image for padding and add text
            padded_image = np.ones((image.shape[0] + caption_padding, image.shape[1], 3), dtype=np.uint8) * 255
            padded_image[caption_padding:, :] = image  # Place the original image below the padding
            cv2.putText(padded_image, caption_text, (text_x, text_y), font, font_scale, text_color, text_thickness)

            return padded_image

        images_with_captions = [
            add_caption_and_padding(image, caption[idx])
            for idx, image in enumerate(resized_images)
        ]

        # Split images into two rows: first 3 images and last 2 images
        row1 = images_with_captions[:3]
        row2 = images_with_captions[3:]

        # Create padded rows
        def create_row(images):
            if not images:
                return None
            # Add padding between images
            padded_images = [
                                np.pad(img, ((0, 0), (0, padding), (0, 0)), mode="constant", constant_values=255)
                                for img in images[:-1]
                            ] + [images[-1]]  # No padding after the last image
            return np.hstack(padded_images)

        row1_padded = create_row(row1)
        row2_padded = create_row(row2)

        # Determine final image dimensions and combine rows
        if row2_padded is not None:
            max_width = max(row1_padded.shape[1], row2_padded.shape[1])
            row1_padded = np.pad(
                row1_padded, ((0, padding), (0, max_width - row1_padded.shape[1]), (0, 0)), mode="constant",
                constant_values=255
            )
            row2_padded = np.pad(
                row2_padded, ((0, 0), (0, max_width - row2_padded.shape[1]), (0, 0)), mode="constant",
                constant_values=255
            )
            final_image = np.vstack((row1_padded, row2_padded))
        else:
            final_image = row1_padded

        # Save the concatenated image
        output_image_path = os.path.join(output_path, f"comparison_{i + 1}.jpg")
        cv2.imwrite(output_image_path, final_image)

    print("Comparison images saved.")

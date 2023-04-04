import argparse
from PIL import Image
import os

# Parse the command-line arguments
parser = argparse.ArgumentParser(description='Split 4K JPEG images into 256x256 pixel JPEG images')
parser.add_argument('--input_dir', help='The input directory of 4K JPEG images')
parser.add_argument('--output_dir', help='The output directory for the 256x256 pixel JPEG images')
parser.add_argument('--output_size', type=int, default=256, help='The size of the output images')
args = parser.parse_args()

# Set the input and output directories
input_dir = args.input_dir
output_dir = args.output_dir

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Loop through each file in the input directory
for filename in os.listdir(input_dir):
    # Check if the file is a JPEG image
    if filename.endswith('.jpg') or filename.endswith('.png'):
        # Open the input image
        input_path = os.path.join(input_dir, filename)
        img = Image.open(input_path)

        # Get the width and height of the input image
        width, height = img.size

        # Calculate the number of output images in the x and y directions
        num_x = width // args.output_size
        num_y = height // args.output_size

        # Loop through each output image
        for i in range(num_x):
            for j in range(num_y):
                # Calculate the coordinates of the top left corner of the output image
                x = i * args.output_size
                y = j * args.output_size

                # Crop the input image to create the output image
                output_img = img.crop((x, y, x + args.output_size, y + args.output_size))

                # Save the output image as a JPEG file in the output directory
                output_filename = f'{os.path.splitext(filename)[0]}_{i}_{j}.png'
                output_path = os.path.join(output_dir, output_filename)
                output_img.save(output_path, 'PNG')

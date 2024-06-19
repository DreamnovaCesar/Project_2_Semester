from PIL import Image

def extract_image_size(file_path):
    with Image.open(file_path) as img:
        width, height = img.size
    return width, height

# Example usage
file_path = r"src\Data\Zervix_Tumor_Segments_Mod\zervix_02_01\zervix_02_01_tumor_001.tif"
width, height = extract_image_size(file_path)
print("Width:", width)
print("Height:", height)

import pandas as pd

def get_file_dimensions(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        num_lines = len(lines)
        if num_lines == 0:
            return 0, 0  # Empty file
        else:
            line_lengths = [len(line.strip().split(',')) for line in lines]
            max_width = max(line_lengths)
            return max_width, num_lines

# Example usage
file_path = r"src\Data\Zervix_Tumor_Segments_Mod\zervix_02_01.txt"  # Change this to the path of your CSV file
width, height = get_file_dimensions(file_path)
print("Width:", width)
print("Height:", height)
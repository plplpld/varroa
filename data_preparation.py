"""Script that intends to prepare data in order to feed the algorithm with it.
It turns real-world images into arrays of defined size
and build polygon masks out of the coordinates of the target."""

import os
import glob
import json

import numpy as np
from PIL import Image, ImageDraw

def build_polygon_mask(coordinates, image_size):
    """
    Build mask out of a polygon's coordinates.

    :params coordinates: the polygon's coordinates
    :params image_size: the size of the image the polygon is in

    :type coordinates: list of tuple of int
    :type image_size: int

    :returns: binary numpy array, the polugon's mask
    """
    width = image_size
    height = image_size

    img = Image.new('L', (width, height), 0)
    ImageDraw.Draw(img).polygon(coordinates, outline=1, fill=1)

    return np.array(img)

def main(polygon_file, image_dir, image_size=256):
    """Main function.
    Process the polygon file to create masks aka target.
    Process the images to create image of the right dimension.

    :params polygon_file: path to the json file containing polygon edges
    :params image_dir: path to the image directory

    :type polygon_file: string
    :type image_dir: string
    """
    # Load the polygon file
    with open(polygon_file, 'r') as f:
        polygons = json.load(f)

    # Scan for jpeg in the image_dir
    images = glob.glob(os.path.join(image_dir, '*.jpg'))
    print("Found {image_number} images in {image_directory}"\
        .format(image_number=len(images),
                image_directory=image_dir))

    for i, image_path in enumerate(images):
        #TODO:split images into smaller part
        #Not sure yet about when the polygons should be done
        #so i'm putting this part on a hiatus.
        #split_and_save_image_and_mask(image_path, polygons, image_size)
        if not i+1 % 10:
            print("{number} images splitted.".format(number=i+1))

    #TODO: add train/test split

if __name__ == '__main__':
    pass

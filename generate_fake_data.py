"""Generation of fake data."""

import numpy as np

def gen_fake_data(n_samples=10, image_size=256):
    """Function that randomly generate images
    containing disks to be detected.
    For each sample the process goes like this:
    1. Randomly pick an integer between 1 and 10 (number of circles)
    2. Randomly generate coordinates, radius and color
    3. Colorise the disks in black

    :params n_samples: number of samples to generate
    :params image_size: height and width of image in pixel

    :type n_samples: int (default: 10)
    :type image_size: int (default: 256)

    :returns: two tensors of shape (n_sample, image_size, image_size, 3)
    """
    # Randomly define the disks to draw per sample
    n_disks = np.random.randint(low=1, high=11, size=n_samples)
    disk_xs = [np.random.randint(0,
                                 image_size,
                                 size=n_disk) for n_disk in n_disks]
    disk_ys = [np.random.randint(0,
                                 image_size,
                                 size=n_disk) for n_disk in n_disks]
    disk_radius = [np.random.randint(2,
                                     image_size // 4,
                                     size=n_disk) for n_disk in n_disks]
    disk_color = [np.random.randint(0, 3, size=n_disk) for n_disk in n_disks]
    disk_data = zip(disk_xs,
                    disk_ys,
                    disk_radius,
                    disk_color)

    # Draw the disks
    def draw_disks(x_disks, y_disks, radiuses, colors, image_size=image_size):
        """Draw disks in a numpy array

        :params x_disks: (int) disks' x coordinate
        :params y_disks: (int) disks' y coordinate
        :params radiuses: (int) disks' radiuses
        :params colors: (int) disks' class
        :params image_size: (int) image height and width
        """
        image = np.zeros((image_size, image_size, 3))

        for xc, yc, r, c in zip(x_disks, y_disks, radiuses, colors):
            x, y = np.ogrid[-xc:image_size-xc, -yc:image_size-yc]
            mask = x*x + y*y <= r*r
            image[mask, c] = 255
        return image, mask

    data, labels = zip(*[draw_disks(x_disks, y_disks, radiuses, colors)\
        for x_disks, y_disks, radiuses, colors in disk_data])

    return np.asarray(data),\
           np.asarray(labels).reshape((n_samples, image_size, image_size, 1))

def save_fake_data():
    """TODO: build a saver to unify the input pipeline."""
    pass

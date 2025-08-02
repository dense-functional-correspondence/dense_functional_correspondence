import torchvision
import numpy as np
import random

from torchvision.transforms import functional as TF
from PIL import Image

def resize_image_mask_box(image, mask, bbox, max_length):
    """
    Resizes an image and its corresponding mask so that the longest side of the image is max_length,
    maintaining the aspect ratio. Also adjusts the bounding box accordingly.

    Args:
        image (PIL.Image): The original image.
        mask (PIL.Image): The binary mask image.
        bbox (tuple): A bounding box (x1, y1, x2, y2).
        max_length (int): The maximum length of the longest side after resizing.

    Returns:
        PIL.Image: The resized image.
        PIL.Image: The resized mask.
        tuple: Adjusted bounding box.
    """
    # Get current dimensions
    width, height = image.size

    # do we have to resize?
    if width == max_length or height == max_length:
        return image, mask, bbox

    # Determine which side is longer and calculate the scaling factor
    if max(width, height) == width:
        scaling_factor = max_length / width
    else:
        scaling_factor = max_length / height

    # Calculate new dimensions
    new_width = int(width * scaling_factor)
    new_height = int(height * scaling_factor)

    # Resize the image and mask
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)
    resized_mask = mask.resize((new_width, new_height), Image.NEAREST)  # Use NEAREST for binary images to avoid interpolation artifacts

    # Adjust the bounding box
    x1, y1, x2, y2 = bbox
    adjusted_bbox = (
        int(x1 * scaling_factor),
        int(y1 * scaling_factor),
        int(x2 * scaling_factor),
        int(y2 * scaling_factor)
    )
    adjusted_bbox = np.array(adjusted_bbox)

    return resized_image, resized_mask, adjusted_bbox

def pad_to_square(image, mask, bbox, random_pad=False):
    """
    Pads the input image and mask to make them square with randomized padding, adjusting the bounding box accordingly.
    
    Args:
        image (PIL.Image): The original image.
        mask (PIL.Image): The mask image.
        bbox (tuple): A bounding box (x1, y1, x2, y2).
    
    Returns:
        PIL.Image: Padded image.
        PIL.Image: Padded mask.
        tuple: Adjusted bounding box.
    """
    width, height = image.size

    # Determine the size to pad to (the maximum of width and height)
    new_size = max(width, height)

    if random_pad:
        # Generate random padding amounts
        # Ensure total padding sums up to new_size - width or new_size - height
        pad_left = random.randint(0, new_size - width)
        pad_top = random.randint(0, new_size - height)
    else:
        # Calculate even padding amounts
        pad_left = (new_size - width) // 2
        pad_top = (new_size - height) // 2

    # Create new images with white background for padding
    new_image = Image.new('RGB', (new_size, new_size), (255, 255, 255))
    new_mask = Image.new('L', (new_size, new_size), 0)  # Assuming mask is a binary image

    # Paste the original images at the randomized offset
    new_image.paste(image, (pad_left, pad_top))
    new_mask.paste(mask, (pad_left, pad_top))

    # Adjust the bounding box
    x1, y1, x2, y2 = bbox
    new_bbox = np.array([x1 + pad_left, y1 + pad_top, x2 + pad_left, y2 + pad_top])

    return new_image, new_mask, new_bbox

def pad_to_square_reflective(image, mask, bbox, random_pad=False):
    """
    Pads the input image and mask to make them square with randomized padding, adjusting the bounding box accordingly.
    
    Args:
        image (PIL.Image): The original image.
        mask (PIL.Image): The mask image.
        bbox (tuple): A bounding box (x1, y1, x2, y2).
    
    Returns:
        PIL.Image: Padded image.
        PIL.Image: Padded mask.
        tuple: Adjusted bounding box.
    """
    width, height = image.size

    # Determine the size to pad to (the maximum of width and height)
    new_size = max(width, height)

    if random_pad:
        # Generate random padding amounts
        # Ensure total padding sums up to new_size - width or new_size - height
        pad_left = random.randint(0, new_size - width)
        pad_top = random.randint(0, new_size - height)
    else:
        # Calculate even padding amounts
        pad_left = (new_size - width) // 2
        pad_top = (new_size - height) // 2
    
    # new_image, https://stackoverflow.com/questions/52471817/performing-a-reflective-center-pad-on-an-image
    new_image = np.array(image)
    new_image = np.pad(new_image, pad_width = ((pad_top, new_size-height-pad_top), (pad_left, new_size-width-pad_left), (0,0)), mode='symmetric')
    new_image = Image.fromarray(new_image)

    # Create new images with white background for padding
    new_mask = Image.new('L', (new_size, new_size), 0)  # Assuming mask is a binary image
    # Paste the original images at the randomized offset
    new_mask.paste(mask, (pad_left, pad_top))
    # Adjust the bounding box
    x1, y1, x2, y2 = bbox
    new_bbox = np.array([x1 + pad_left, y1 + pad_top, x2 + pad_left, y2 + pad_top])

    return new_image, new_mask, new_bbox

class HorizontalFlip(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img, bbox):
        if np.random.rand() < self.p:
            W, H = img.size

            img = TF.hflip(img)

            # vertically flip the pixel coordinates for each set of UVs
            x0, y0, x1, y1 = bbox

            x0_new = (W-1) - x1
            x1_new = (W-1) - x0

            bbox = (x0_new,y0,x1_new,y1)

            return img, bbox
        else:
            return img, bbox



class RandomCrop(object):
    def __init__(self):
        self.m = 10 #margin pixels

    def __call__(self, img, mask, bbox):
        x0, y0, x1, y1 = bbox

        w, h = img.size
        x_c = (x0+x1)//2
        y_c = (y1+y0)//2

        bbox_h = y1 - y0
        bbox_w = x1 - x0

        try:
            min_crop_size = max(bbox_h, bbox_w) + self.m
            max_crop_size = min(h,w)

            crop_size = random.randint(min_crop_size, max_crop_size)
        
            # how much can we shift in either direction
            max_shift_x = (crop_size - bbox_w - self.m)//2 - 1
            max_shift_y = (crop_size - bbox_h - self.m)//2 - 1

            y_shift = np.random.randint(-max_shift_y, max_shift_y) if max_shift_y != 0 else 0
            x_shift = np.random.randint(-max_shift_x, max_shift_x) if max_shift_x != 0 else 0
        except Exception as e:
            #print("Exception", e, "in loader") 
            return img, mask, bbox

        y_min = (y_c + y_shift) - crop_size//2
        x_min = (x_c + x_shift) - crop_size//2
            
        img = torchvision.transforms.functional.crop(img, y_min, x_min, crop_size, crop_size)
        mask = torchvision.transforms.functional.crop(mask, y_min, x_min, crop_size, crop_size)

        x0 -= x_min
        x1 -= x_min
        y0 -= y_min
        y1 -= y_min

        bbox = x0, y0, x1, y1

        return img, mask, bbox

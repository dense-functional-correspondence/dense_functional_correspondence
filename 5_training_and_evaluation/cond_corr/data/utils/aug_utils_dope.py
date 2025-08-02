import torch
import os
import numpy as np
import cv2
import copy
import json
import random
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image
from scipy import ndimage

class ContrastiveAugmentation(object):
    def __init__(self, param_dict):
        '''
            takes the augmentation_settings.json file and 
            defines a set of data augmentations based on the parameters
            defined for each type of data augmentation

            the data augmentation functions themselves are defined as classes
            below in this file
        '''

        self.hflip_aug = None
        self.vflip_aug = None
        self.brightness_aug = None
        self.hue_aug = None
        self.saturation_aug = None
        self.gamma_aug = None
        self.contrast_aug = None
        self.blur_aug = None
        self.random_masking_aug = None

        aug_list = list(param_dict.keys())
        
        if "HorizontalFlip" in aug_list:    
            self.hflip_aug = HorizontalFlip(*param_dict['HorizontalFlip'])

        if "VerticalFlip" in aug_list:
            self.vflip_aug = VerticalFlip(*param_dict['VerticalFlip'])

        if "BrightnessAug" in aug_list:
            self.brightness_aug = BrightnessAug(*param_dict['BrightnessAug'])
        
        if "HueAug" in aug_list:
            self.hue_aug = HueAug(*param_dict['HueAug'])

        if "GammaAug" in aug_list:
            self.gamma_aug = GammaAug(*param_dict['GammaAug'])

        if "ContrastAug" in aug_list:
            self.contrast_aug = ContrastAug(*param_dict['ContrastAug'])

        if "SaturationAug" in aug_list:
            self.saturation_aug = SaturationAug(*param_dict['SaturationAug'])

        if "BlurAug" in aug_list:
            self.blur_aug = BlurAug(*param_dict['BlurAug'])

        if "RandomMaskingAug" in aug_list:
            self.random_masking_aug = RandomMaskingAug(*param_dict['RandomMaskingAug'])

    def __call__(self, img, mask, label, uv_ls):
        
        if uv_ls is not None:
            uv_ls = uv_ls[:,[1,0]] # uv_ls should be row-column indices
            
        # image augmentations
        if self.hue_aug is not None:
            img = self.hue_aug(img)

        if self.saturation_aug is not None:
            img = self.saturation_aug(img)

        if self.brightness_aug is not None:
            img = self.brightness_aug(img)
        
        if self.gamma_aug is not None:
            img = self.gamma_aug(img)

        if self.contrast_aug is not None:
            img = self.contrast_aug(img)
    
        if self.blur_aug is not None:
            img = self.blur_aug(img)

        if self.hflip_aug is not None:
            img, mask, label, uv_ls = self.hflip_aug(img, mask, label, uv_ls)
        
        if self.vflip_aug is not None:
            img, mask, label, uv_ls = self.vflip_aug(img, mask, label, uv_ls)

        if self.random_masking_aug is not None:
            img, mask, label, uv_ls = self.random_masking_aug(img, mask, label, uv_ls)
        
        H, W = img.size
        
        if uv_ls is not None:
            # making sure we don't get out of bounds errors
            uv_ls[uv_ls<0]=0
            uv_ls[uv_ls>(W-1)]=W-1
            uv_ls = uv_ls[:,[1,0]] # return column-row
          
        return img, mask, label, uv_ls


class VerticalFlip(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img, mask, label=None, uv_list=None):
        if np.random.rand() < self.p:
            H, W = img.size

            img = TF.vflip(img)
            mask = TF.vflip(mask)
            
            if label is not None:
                label = TF.vflip(label)

            if uv_list is not None:
                uv_list[:,0] = (H-1) - uv_list[:,0]

        return img, mask, label, uv_list

class HorizontalFlip(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img, mask, label=None, uv_list=None):
        if np.random.rand() < self.p:
            H, W = img.size

            img = TF.hflip(img)
            mask = TF.hflip(mask)

            if label is not None:
                label = TF.hflip(label)

            if uv_list is not None:
                # vertically flip the pixel coordinates for each set of UVs
                uv_list[:,1] = (W-1) - uv_list[:,1]
    
        return img, mask, label, uv_list

class BrightnessAug(object):
    def __init__(self, p, min_val, max_val):
        self.p = p
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, img):
        if np.random.rand() < self.p:
            brightness = np.random.uniform(self.min_val, self.max_val)
            img = transforms.functional.adjust_brightness(img, brightness)

        return img

class HueAug(object):
    def __init__(self, p, min_val, max_val):
        self.p = p
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, img):
        if np.random.rand() < self.p:
            hue = np.random.uniform(self.min_val, self.max_val)
            img = transforms.functional.adjust_hue(img, hue)

        return img

class GammaAug(object):
    def __init__(self, p, min_val, max_val):
        self.p = p
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, img):
        if np.random.rand() < self.p:
            gamma = np.random.uniform(self.min_val, self.max_val)
            img = transforms.functional.adjust_gamma(img, gamma)

        return img

class ContrastAug(object):
    def __init__(self, p, min_val, max_val):
        self.p = p
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, img):
        if np.random.rand() < self.p:
            contrast = np.random.uniform(self.min_val, self.max_val)
            img = transforms.functional.adjust_contrast(img, contrast)

        return img

class SaturationAug(object):
    def __init__(self, p, min_val, max_val):
        self.p = p
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, img):
        if np.random.rand() < self.p:
            saturation = np.random.uniform(self.min_val, self.max_val)
            img = transforms.functional.adjust_saturation(img, saturation)

        return img

class BlurAug(object):
    def __init__(self, p, kernels, sigma_min, sigma_max):

        self.p = p
        self.kernels = kernels
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, img):
        if np.random.rand() < self.p:
            kernel_size = int(np.random.choice(self.kernels))
            sigma = np.random.uniform(self.sigma_min, self.sigma_max)
            img = transforms.functional.gaussian_blur(img, kernel_size, sigma)

        return img


# Geometric and BG Augmentations!!
class RandomMaskingAug(object):
    def __init__(self, p, geometric_p, background, aug_ls):

        assert background in ["domain_randomized", "blank"] 

        self.p = p
        self.geo_p = geometric_p
        self.background = background
        self.aug_ls = aug_ls

    def __call__(self, img, mask, label=None, uv_ls=None):
        
        if np.random.rand() < self.p:
            img = np.asarray(img)
            mask = np.asarray(mask) // 255
            if label is not None:
                label = np.asarray(label)

            if np.random.rand() < self.geo_p:
                # try:
                img, mask, label, uv_ls = do_geometric_augmentation(img, mask, label, uv_ls, self.aug_ls)
                # except Exception:
                #     print(Exception)
                #     pass

                img = img * mask
            else:
                img = img * mask

            if self.background == "blank":
                img = Image.fromarray(img)
                mask = Image.fromarray((mask * 255).astype(np.uint8))
            
            if self.background == "domain_randomized":
                img = domain_randomize_background(img, mask)
                img = Image.fromarray(img)
                mask = Image.fromarray((mask * 255).astype(np.uint8))
            
            if label is not None:
                label = Image.fromarray(label)

        return img, mask, label, uv_ls


def perspective(image, mask, label=None, uv=None):
    
    H, W, C = image.shape

    pts1 = np.float32([[0,0], [W,0],
                       [0,H],[W,H]])
    p_W = np.random.randint(-0.3*W, 0.3*W, size=4)
    p_H = np.random.randint(-0.3*H, 0.3*H, size=4)
    pts2 = np.float32([[0+p_W[0],0+p_H[0]], [W+p_W[1],0+p_H[1]],
                       [0+p_W[2],H+p_H[2]], [W+p_W[3],H+p_H[3]]])

    s = np.max([np.abs(p_W).max(), np.abs(p_H).max()])
    pts2+=s

    M = cv2.getPerspectiveTransform(pts1, pts2)

    if uv is not None:
      uv_ = uv[:,::-1]
      x_new = (M[0,0]*uv_[:,0] + M[0,1]*uv_[:,1] + M[0,2]) / (M[2,0]*uv_[:,0] + M[2,1]*uv_[:,1] + M[2,2])
      y_new = (M[1,0]*uv_[:,0] + M[1,1]*uv_[:,1] + M[1,2]) / (M[2,0]*uv_[:,0] + M[2,1]*uv_[:,1] + M[2,2])
      uv_new = np.stack([y_new, x_new]).T

    t_img = cv2.warpPerspective(image, M, (3*W, 3*H))
    t_mask = cv2.warpPerspective(mask, M, (3*W, 3*H))
    if label is not None:
        t_label = cv2.warpPerspective(label, M, (3*W, 3*H))

    xmin, xmax, ymin, ymax = get_bbox(t_mask)

    t_img = t_img[ymin:ymax, xmin:xmax,:]
    t_mask = t_mask[ymin:ymax, xmin:xmax,:]
    if label is not None:
        t_label = t_label[ymin:ymax, xmin:xmax,:]
    else:
        t_label = None

    if uv is not None:
        uv_new[:,0] -= ymin
        uv_new[:,1] -= xmin
        uv_new = (uv_new).astype(int)
    else:
        uv_new = None
    
    return t_img, t_mask, t_label, uv_new

def rotation(image, mask, label=None, uv=None, angle=None):
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping
    """

    height, width = image.shape[:2] # image shape has 3 dimensions
    image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0,0])
    abs_sin = abs(rotation_mat[0,1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    image = cv2.warpAffine(image, rotation_mat, (bound_w, bound_h))
    mask = cv2.warpAffine(mask, rotation_mat, (bound_w, bound_h))
    if label is not None:
        label = cv2.warpAffine(label, rotation_mat, (bound_w, bound_h))

    if uv is not None:
        x_new = rotation_mat[0,0]*uv[:,1] + rotation_mat[0,1]*uv[:,0] + rotation_mat[0,2]
        y_new = rotation_mat[1,0]*uv[:,1] + rotation_mat[1,1]*uv[:,0] + rotation_mat[1,2]
        uv = np.stack([y_new, x_new]).T

    return image, mask, label, uv

def scaling(image, mask, label=None, uv=None):

    # min_side_len = 225*(1/4)
    min_side_len = 225*(5/8) # Don't make it so small!
    max_side_len = 225*(7/8)

    H, W, _ = image.shape
    longest_side = np.max([H,W])
    new_longest_side = np.random.randint(min_side_len, max_side_len)

    factor = new_longest_side / longest_side
    factor = np.min([factor, 1.5])

    image = cv2.resize(image, (int(W*factor), int(H*factor)))
    mask = cv2.resize(mask, (int(W*factor), int(H*factor)), interpolation = cv2.INTER_NEAREST)
    if label is not None:
        label = cv2.resize(label, (int(W*factor), int(H*factor)), interpolation = cv2.INTER_NEAREST)
    
    if uv is not None:
        uv = (uv*factor).astype(int)

    return image, mask, label, uv

def place_patch_in_image(image_patch, mask_patch, label_patch=None, uv=None, im_size=224):

    patch_H, patch_W, _ = image_patch.shape
    return_im = np.zeros((im_size, im_size, 3), dtype=np.uint8)
    return_mask = np.zeros((im_size, im_size, 3), dtype=np.uint8)
    if label_patch is not None:
        return_label = np.zeros((im_size, im_size, 3), dtype=np.uint8)

    T_max_H = im_size - patch_H - 10
    T_max_W = im_size - patch_W - 10

    T_H = np.random.randint(T_max_H)
    T_W = np.random.randint(T_max_W)

    assert T_H >= 0
    assert T_W >= 0

    return_im[T_H:T_H+patch_H, T_W:T_W+patch_W,:] = image_patch
    return_mask[T_H:T_H+patch_H, T_W:T_W+patch_W,:] = mask_patch
    if label_patch is not None:
        return_label[T_H:T_H+patch_H, T_W:T_W+patch_W,:] = label_patch
    else:
        return_label = None

    if uv is not None:
        uv[:,0] += T_H
        uv[:,1] += T_W

    return return_im, return_mask, return_label, uv

def do_geometric_augmentation(image, mask, label=None, uv=None, aug_list=None):

    xmin, xmax, ymin, ymax = get_bbox(mask)

    test_im = np.zeros((224,224,3)).astype(np.float32)
    test_im[0:5,:,:] = 1
    test_im[-5:,:,:] = 1
    test_im[:,0:5,:] = 1
    test_im[:,-5:,:] = 1
    
    output = mask * test_im
    bad = np.any(output)
    
    if not bad:
        # cropping
        image = image[ymin:ymax, xmin:xmax, :]
        mask = mask[ymin:ymax, xmin:xmax, :]
        if label is not None:
            label = label[ymin:ymax, xmin:xmax, :]

        if uv is not None:
            uv[:,0] -= ymin
            uv[:,1] -= xmin
        
        if np.random.rand() < 0.5: # do these operations less...
            # rotating patch
            if "rotation" in aug_list:
                deg = np.random.randint(0,360)
                image, mask, label, uv = rotation(image, mask, label, uv, deg)
            # perspective transform
            if "perspective" in aug_list:
                image, mask, label, uv = perspective(image, mask, label, uv)
            # scaling patch
            if "scaling" in aug_list:
                image, mask, label, uv = scaling(image, mask, label, uv)
        # compositing patch in image
        image, mask, label, uv = place_patch_in_image(image, mask, label, uv)

    return image, mask, label, uv

def domain_randomize_background(image_rgb, image_mask):
    # First, mask the rgb image
    image_rgb_numpy = np.asarray(image_rgb)
    image_mask_numpy = np.asarray(image_mask)
    three_channel_mask = image_mask_numpy

    # Next, domain randomize all non-masked parts of image
    three_channel_mask_complement = np.ones_like(three_channel_mask) - three_channel_mask
    random_rgb_image = get_random_image(image_rgb_numpy.shape)
    random_rgb_background = three_channel_mask_complement * random_rgb_image

    domain_randomized_image_rgb = image_rgb_numpy + random_rgb_background
    return domain_randomized_image_rgb

def get_random_image(shape):
    """
    Expects something like shape=(480,640,3)
    :param shape: tuple of shape for numpy array, for example from my_array.shape
    :type shape: tuple of ints
    :return random_image:
    :rtype: np.ndarray
    """
    if random.random() < 0.5:
        rand_image = get_random_solid_color_image(shape)
    else:
        rgb1 = get_random_solid_color_image(shape)
        rgb2 = get_random_solid_color_image(shape)
        vertical = bool(np.random.uniform() > 0.5)
        rand_image = get_gradient_image(rgb1, rgb2, vertical=vertical)

    if random.random() < 0.5:
        return rand_image
    else:
        return add_noise(rand_image)

def get_random_rgb():
    """
    :return random rgb colors, each in range 0 to 255, for example [13, 25, 255]
    :rtype: numpy array with dtype=np.uint8
    """
    return np.array(np.random.uniform(size=3) * 255, dtype=np.uint8)

def get_random_solid_color_image(shape):
    """
    Expects something like shape=(480,640,3)
    :return random solid color image:
    :rtype: numpy array of specificed shape, with dtype=np.uint8
    """
    return np.ones(shape,dtype=np.uint8)*get_random_rgb()

def get_random_entire_image(shape, max_pixel_uint8):
    """
    Expects something like shape=(480,640,3)
    Returns an array of that shape, with values in range [0..max_pixel_uint8)
    :param max_pixel_uint8: maximum value in the image
    :type max_pixel_uint8: int
    :return random solid color image:
    :rtype: numpy array of specificed shape, with dtype=np.uint8
    """
    return np.array(np.random.uniform(size=shape) * max_pixel_uint8, dtype=np.uint8)

# this gradient code roughly taken from:
# https://github.com/openai/mujoco-py/blob/master/mujoco_py/modder.py
def get_gradient_image(rgb1, rgb2, vertical):
    """
    Interpolates between two images rgb1 and rgb2
    :param rgb1, rgb2: two numpy arrays of shape (H,W,3)
    :return interpolated image:
    :rtype: same as rgb1 and rgb2
    """
    bitmap = np.zeros_like(rgb1)
    h, w = rgb1.shape[0], rgb1.shape[1]
    if vertical:
        p = np.tile(np.linspace(0, 1, h)[:, None], (1, w))
    else:
        p = np.tile(np.linspace(0, 1, w), (h, 1))

    for i in range(3):
        bitmap[:, :, i] = rgb2[:, :, i] * p + rgb1[:, :, i] * (1.0 - p)

    return bitmap

def add_noise(rgb_image):
    """
    Adds noise, and subtracts noise to the rgb_image
    :param rgb_image: image to which noise will be added
    :type rgb_image: numpy array of shape (H,W,3)
    :return image with noise:
    :rtype: same as rgb_image
    ## Note: do not need to clamp, since uint8 will just overflow -- not bad
    """
    max_noise_to_add_or_subtract = 50
    return rgb_image + get_random_entire_image(rgb_image.shape, max_noise_to_add_or_subtract) - get_random_entire_image(rgb_image.shape, max_noise_to_add_or_subtract)

def get_bbox(img):

    img = np.array(img).copy()
    img = ndimage.binary_dilation(img, iterations=2).astype(np.uint8)

    horz = np.sum(img, axis=0)
    vert = np.sum(img, axis=1)

    horz = np.where(horz >= 1)
    vert = np.where(vert >= 1)

    x_min = horz[0][0]
    x_max = horz[0][-1]
    y_min = vert[0][0]
    y_max = vert[0][-1]

    return [x_min, x_max, y_min, y_max]
import kornia
import os
import cv2
import kornia as K
import kornia.feature as KF
import kornia.image
import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
from kornia.feature.adalam import AdalamFilter
from kornia_moons.viz import *
from kornia_moons.feature import *
import sklearn.metrics
import math

from typing import Any, Dict

from distance_matcher import detect_keypoints_and_match, \
    detect_keypoints_and_descriptors_knn_match, \
        detect_keypoints_and_match_SIFT, \
            detect_keypoints_and_match_BRISK
import utils as utils

#######################################################################
'''Utility functions'''
#######################################################################

# https://kornia.github.io/tutorials/nbs/image_matching_lightglue.html
def get_matching_keypoints(kp1, kp2, idxs):
    mkpts1 = kp1[idxs[:, 0]]
    mkpts2 = kp2[idxs[:, 1]]
    #print(len(mkpts1), len(mkpts2))
    return mkpts1, mkpts2

# Convert tensor to binary tensor (for hamming distance)
def binary(x, bits):
    mask = 2**torch.arange(bits).to(x.device, x.dtype)
    res = x.unsqueeze(-1).bitwise_and(mask).ne(0).byte()
    return torch.reshape(res, (res.size(dim=0), -1))

#######################################################################
'''
Testing different kornia matchers

For a test on the different matching options, we use DISK for feature 
detection if the matcher allows for this matcher.
All these matching functions can be seen at 
https://kornia.readthedocs.io/en/latest/feature.html#matching matching 
tab. The functions are named after the matcher used.'''
#######################################################################

''' Nearest neighbour matching using DISK'''
def kornia_matcher_test_nn(img1_name: str, img2_name: str, 
                           device: torch.device, 
                           feature_extractor: KF.DISK) -> tuple:
    # Load images from the given paths
    img1 = K.io.load_image(img1_name, K.io.ImageLoadType.RGB32, 
                           device=device)[None, ...]
    img2 = K.io.load_image(img2_name, K.io.ImageLoadType.RGB32, 
                           device=device)[None, ...]

    with torch.inference_mode():
        # Concatinate images
        inp = torch.cat([img1, img2], dim=0)

        # Use the feature_extractor, disk as the standard, to detect
        # the features of the concatinated image
        num_features = 128
        features1, features2 = feature_extractor(
            inp, num_features, pad_if_not_divisible=True)
        kps1, descs1 = features1.keypoints, features1.descriptors
        kps2, descs2 = features2.keypoints, features2.descriptors

        # Compute nearest neighbour matching
        dists, idxs = KF.match_nn(descs1, descs2)
    
    return dists, img1, kps1, img2, kps2

''' Mutual nearest neighbour matching using DISK'''
def kornia_matcher_test_mnn(img1_name: str, img2_name: str, 
                            device: torch.device, 
                           feature_extractor: KF.DISK) -> tuple:
    # Load images from the given paths
    img1 = K.io.load_image(img1_name, K.io.ImageLoadType.RGB32, 
                           device=device)[None, ...]        
    img2 = K.io.load_image(img2_name, K.io.ImageLoadType.RGB32, 
                           device=device)[None, ...]

    with torch.inference_mode():
        # Concatinate images
        inp = torch.cat([img1, img2], dim=0)

        # Use the feature_extractor, disk as the standard, to detect 
        # the features of the concatinated image
        num_features = 128
        features1, features2 = feature_extractor(
            inp, num_features, pad_if_not_divisible=True)
        kps1, descs1 = features1.keypoints, features1.descriptors
        kps2, descs2 = features2.keypoints, features2.descriptors

        # Compute mutual nearest neighbourmatching
        dists, idxs = KF.match_mnn(descs1, descs2)
    
    return dists, img1, kps1, img2, kps2

'''Nerest neighbour distance ratio matching using DISK'''
def kornia_matcher_test_snn(img1_name: str, img2_name: str, 
                            device: torch.device, 
                           feature_extractor: KF.DISK) -> tuple:
    # Load images from the given paths
    img1 = K.io.load_image(img1_name, K.io.ImageLoadType.RGB32, 
                           device=device)[None, ...]        
    img2 = K.io.load_image(img2_name, K.io.ImageLoadType.RGB32, 
                           device=device)[None, ...]

    with torch.inference_mode():
        # Concatinate images
        inp = torch.cat([img1, img2], dim=0)

        # Use the feature_extractor, disk as the standard, to detect 
        # the features of the concatinated image
        num_features = 128
        features1, features2 = feature_extractor(
            inp, num_features, pad_if_not_divisible=True)
        kps1, descs1 = features1.keypoints, features1.descriptors
        kps2, descs2 = features2.keypoints, features2.descriptors

        # Compute matching with Lowe's ratio test
        dists, idxs = KF.match_snn(descs1, descs2, th=0.85)
    
    return dists, img1, kps1, img2, kps2

''' Mutual nearest neighbour distance ratio matching using DISK.
It also offers options to match using Hamming distances or to compute
a matching for images with different sizes (smaller than 224x224)'''
def kornia_matcher_test_smnn(img1_name: str, img2_name: str, 
                             device: torch.device, 
                             feature_extractor: KF.DISK, 
                             th: float = 0.85, hamming: bool = False, 
                             const_size_set = False ) -> tuple:
    # Load images from the given paths
    img1 = K.io.load_image(img1_name, K.io.ImageLoadType.RGB32, 
                           device=device)[None, ...]        
    img2 = K.io.load_image(img2_name, K.io.ImageLoadType.RGB32, 
                           device=device)[None, ...]

    with torch.inference_mode():
        # Special case to concatinate images of different sizes
        # (fill the missing pixels up with black)
        if(const_size_set):
             # MAXIMUM SIZE OF THE IMAGES SET HERE (224 standard)
            new_img_h = new_img_w = 224

            diffh_1, diffw_1 = new_img_h-img1.size(2), \
                new_img_w-img1.size(3)
            diffh_2, diffw_2 = new_img_h-img2.size(2), \
                new_img_w-img2.size(3)

            # Pad missing pixels evenly
            img1 = torch.nn.functional.pad(input=img1, 
                                           pad=(math.ceil(diffw_1/2), 
                                                math.floor(diffw_1/2), 
                                                math.ceil(diffh_1/2), 
                                                math.floor(diffh_1/2)), 
                                                mode='constant', 
                                                value=0)
            
            img2 = torch.nn.functional.pad(input=img2, 
                                           pad=(math.ceil(diffw_2/2), 
                                                math.floor(diffw_2/2), 
                                                math.ceil(diffh_2/2), 
                                                math.floor(diffh_2/2)), 
                                                mode='constant', 
                                                value=0)
        # Concatinate images
        inp = torch.cat([img1, img2], dim=0)

        # Use the feature_extractor, disk as the standard, to detect 
        # the features of the concatinated image
        num_features = 128
        features1, features2 = feature_extractor(
            inp, num_features, pad_if_not_divisible=True)
        kps1, descs1 = features1.keypoints, features1.descriptors
        kps2, descs2 = features2.keypoints, features2.descriptors

        # Compute matching
        # Hamming distance computation just for testing, works very bad
        if hamming:
            max = torch.max(torch.max(kps1), torch.max(kps2)).item()
            hamming_dists = torch.from_numpy(
                sklearn.metrics.pairwise_distances(
                    binary(kps1.cpu().to(dtype=torch.uint8), max), 
                    binary(kps2.cpu().to(dtype=torch.uint8), max), 
                    metric='hamming'))

            dists, idxs = KF.match_smnn(descs1, descs2, th=th, 
                                        dm=hamming_dists)
        else:
            dists, idxs = KF.match_smnn(descs1, descs2, th=th)
    
    return dists, img1, kps1, img2, kps2

''' Mutual nearest neighbour distance ratio matching using DISK with 
USAC_ACCURATE estimator for outlier detection.'''
def kornia_matcher_test_smnn_estimator(img1_name: str, img2_name: str,
                                        device: torch.device, 
                                        feature_extractor: KF.DISK, 
                                        th: float = 0.85) -> tuple:
    # Load images from the given paths
    img1 = K.io.load_image(img1_name, K.io.ImageLoadType.RGB32, 
                           device=device)[None, ...]        
    img2 = K.io.load_image(img2_name, K.io.ImageLoadType.RGB32, 
                           device=device)[None, ...]

    with torch.inference_mode():
        # Concatinate images
        inp = torch.cat([img1, img2], dim=0)

        # Use the feature_extractor, disk as the standard, to detect 
        # the features of the concatinated image
        num_features = 128
        features1, features2 = feature_extractor(
            inp, num_features, pad_if_not_divisible=True)
        kps1, descs1 = features1.keypoints, features1.descriptors
        kps2, descs2 = features2.keypoints, features2.descriptors

        # Compute matching
        dists, idxs = KF.match_smnn(descs1, descs2, th=th)
        
        mkpts1, mkpts2 = get_matching_keypoints(kps1, kps2, idxs)
        
    # Applying the estimator
    Fm, inliers = cv2.findFundamentalMat(
            kps1.detach().cpu().numpy(), 
            kps2.detach().cpu().numpy(), 
            cv2.USAC_ACCURATE, 1.0, 0.999, 100000)

    inliers = inliers > 0
    
    return dists, img1, kps1, img2, kps2

''' First to second geometrically inconsistend nearest neighbour using
DISK.'''
def kornia_matcher_test_fginn(img1_name: str, img2_name: str, 
                              device: torch.device, 
                              feature_extractor: KF.DISK) -> tuple:
    # Load images from the given paths
    img1 = K.io.load_image(img1_name, K.io.ImageLoadType.RGB32, 
                           device=device)[None, ...]        
    img2 = K.io.load_image(img2_name, K.io.ImageLoadType.RGB32, 
                           device=device)[None, ...]

    with torch.inference_mode():
        # Concatinate images
        inp = torch.cat([img1, img2], dim=0)

        # Use the feature_extractor, disk as the standard, to detect 
        # the features of the concatinated image
        num_features = 128
        features1, features2 = feature_extractor(
            inp, num_features, pad_if_not_divisible=True)
        kps1, descs1 = features1.keypoints, features1.descriptors
        kps2, descs2 = features2.keypoints, features2.descriptors

        # Computs LAFs for fginn
        lafs1 = KF.laf_from_center_scale_ori(
            kps1[None], torch.ones(1, len(kps1), 1, 1, device=device))
        lafs2 = KF.laf_from_center_scale_ori(
            kps2[None], torch.ones(1, len(kps2), 1, 1, device=device))

        # Compute matching
        dists, idxs = KF.match_fginn(descs1, descs2, lafs1, lafs2, 
                                     th=0.85)
    
    return dists, img1, kps1, img2, kps2

''' AdaLAM with near standard configuration. Using again DISK.'''
def kornia_matcher_test_adalam(img1_name: str, img2_name: str, 
                               device: torch.device, 
                               feature_extractor: KF.DISK) -> tuple:
    # Load images from the given paths
    img1 = K.io.load_image(img1_name, K.io.ImageLoadType.RGB32, 
                           device=device)[None, ...]        
    img2 = K.io.load_image(img2_name, K.io.ImageLoadType.RGB32, 
                           device=device)[None, ...]

    hw1 = torch.tensor(img1.shape[2:], device=device)
    hw2 = torch.tensor(img2.shape[2:], device=device)

    adalam_config = KF.adalam.get_adalam_default_config()
    adalam_config["force_seed_mnn"] = False
    adalam_config["search_expansion"] = 16
    adalam_config["ransac_iters"] = 256
    #adalam_config["device"] = device

    num_features = 2048

    with torch.inference_mode():
        # Concatinate images
        inp = torch.cat([img1, img2], dim=0)
        features1, features2 = \
            feature_extractor(inp, num_features, 
                              pad_if_not_divisible=True)
        kps1, descs1 = features1.keypoints, features1.descriptors
        kps2, descs2 = features2.keypoints, features2.descriptors

        # Compute LAFs
        lafs1 = KF.laf_from_center_scale_ori(
            kps1[None], 96 * torch.ones(1, len(kps1), 1, 1, 
                                        device=device))
        lafs2 = KF.laf_from_center_scale_ori(
            kps2[None], 96 * torch.ones(1, len(kps2), 1, 1, 
                                        device=device))

        # Compute matching
        dists, idxs = KF.match_adalam(descs1, descs2, lafs1, lafs2, 
                                      hw1=hw1, hw2=hw2, 
                                      config=adalam_config)
        
    return dists, img1, descs1, img2, descs2

''' LightGlue matcher using DISK'''
def kornia_matcher_test_lightglue(img1_name: str, img2_name: str, 
                                  device: torch.device, 
                                  feature_extractor: KF.DISK, 
                                  matcher: KF.LightGlueMatcher
                                  ) -> tuple:
    # Load images from the given paths
    img1 = K.io.load_image(img1_name, K.io.ImageLoadType.RGB32, 
                           device=device)[None, ...]        
    img2 = K.io.load_image(img2_name, K.io.ImageLoadType.RGB32, 
                           device=device)[None, ...]

    num_features = 128

    hw1 = torch.tensor(img1.shape[2:], device=device)
    hw2 = torch.tensor(img2.shape[2:], device=device)

    with torch.no_grad():
        # Concatinate images
        inp = torch.cat([img1, img2], dim=0)
        features1, features2 = feature_extractor(
            inp, num_features, pad_if_not_divisible=True)
        kps1, descs1 = features1.keypoints, features1.descriptors
        kps2, descs2 = features2.keypoints, features2.descriptors

        # Compute LAFs
        lafs1 = KF.laf_from_center_scale_ori(
            kps1[None], torch.ones(
                1, len(kps1), 1, 1, device=device))
        lafs2 = KF.laf_from_center_scale_ori(
            kps2[None], torch.ones(
                1, len(kps2), 1, 1, device=device))
    
        # Computing matching
        dists, idxs = matcher(descs1, descs2, lafs1, 
                            lafs2, hw1=hw1, hw2=hw2)

    return dists, img1, kps1, img2, kps2, idxs

'''LoFTR matcher, including keypoint detection and description.'''
def kornia_matcher_test_LoFTR(img1_name, img2_name, device, matcher):
    img1 = K.io.load_image(img1_name, K.io.ImageLoadType.RGB32, 
                           device=device)[None, ...]
    img2 = K.io.load_image(img2_name, K.io.ImageLoadType.RGB32, 
                           device=device)[None, ...]

    # LofTR works on grayscale images only
    input_dict = {
        "image0": K.color.rgb_to_grayscale(img1),
        "image1": K.color.rgb_to_grayscale(img2),
    }

    with torch.inference_mode():
        correspondences = matcher(input_dict)
    
    mkpts0 = correspondences["keypoints0"].cpu().numpy()
    mkpts1 = correspondences["keypoints1"].cpu().numpy()
    conf = correspondences["confidence"].cpu().numpy()

    return conf, img1, mkpts0, img2, mkpts1


#######################################################################
#
# END OF: Testing different kornia matchers 
#
# Testing the usage of kornia Detectors
#
#######################################################################

''' Computing Shi-Tomasi cornerness function on the images before
using DISK and smnn for matching'''
def kornia_detector_test_gftt_response(img1_name, img2_name, device, 
                                       feature_extractor):
    # Load images from the given paths
    img1 = K.io.load_image(img1_name, K.io.ImageLoadType.RGB32, 
                           device=device)[None, ...]        
    img2 = K.io.load_image(img2_name, K.io.ImageLoadType.RGB32, 
                           device=device)[None, ...]

    # Applying Detectors
    img1 = kornia.feature.gftt_response(
        img1, grads_mode='sobel', sigmas=None)
    img2 = kornia.feature.gftt_response(
        img2, grads_mode='sobel', sigmas=None)

    with torch.inference_mode():
        # Concatinate images
        inp = torch.cat([img1, img2], dim=0)

        # Use the feature_extractor, disk as the standard, to detect 
        # the features of the concatinated image
        num_features = 128
        features1, features2 = feature_extractor(
            inp, num_features, pad_if_not_divisible=True)
        kps1, descs1 = features1.keypoints, features1.descriptors
        kps2, descs2 = features2.keypoints, features2.descriptors

        # Compute matching
        dists, idxs = KF.match_smnn(descs1, descs2, th=0.85)
    
    return dists, img1, kps1, img2, kps2

''' Computing the Difference-of-Gaussian response function on the images 
before using DISK and smnn for matching'''
def kornia_detector_test_dog_response_single(
        img1_name, img2_name, device, feature_extractor):
    # Load images from the given paths
    img1 = K.io.load_image(img1_name, K.io.ImageLoadType.RGB32, 
                           device=device)[None, ...]        
    img2 = K.io.load_image(img2_name, K.io.ImageLoadType.RGB32, 
                           device=device)[None, ...]

    # Applying Detectors
    img1 = kornia.feature.dog_response_single(img1, sigma1=1.0, 
                                              sigma2=1.6)
    img1 = kornia.feature.dog_response_single(img2, sigma1=1.0, 
                                              sigma2=1.6)

    with torch.inference_mode():
        # Concatinate images
        inp = torch.cat([img1, img2], dim=0)

        # Use the feature_extractor, disk as the standard, to detect 
        # the features of the concatinated image
        num_features = 128
        features1, features2 = feature_extractor(
            inp, num_features, pad_if_not_divisible=True)
        kps1, descs1 = features1.keypoints, features1.descriptors
        kps2, descs2 = features2.keypoints, features2.descriptors

        # Compute matching
        dists, idxs = KF.match_smnn(descs1, descs2, th=0.85)
    
    return dists, img1, kps1, img2, kps2

#######################################################################
#
# END OF: Testing the usage of kornia Detectors
#
# Testing the usage of kornia Descriptors
#
#######################################################################

'''Computing DenseSIFT Descriptors on the images before using the
nearest neighbour function for matching'''
def kornia_matcher_test_descriptor_DenseSIFTDescriptor(
        img1_name, img2_name, device, feature_descriptor):
    # Load images from the given paths
    img1 = K.io.load_image(img1_name, K.io.ImageLoadType.GRAY32, 
                           device=device)[None, ...]        
    img2 = K.io.load_image(img2_name, K.io.ImageLoadType.GRAY32, 
                           device=device)[None, ...]

    with torch.inference_mode():
        # Concatinate images
        inp = torch.cat([img1, img2], dim=0)

        # Use the feature_descriptor given through the arguments
        features1, features2 = feature_descriptor(inp)
        descs1 = torch.flatten(features1, 1, 2)
        descs2 = torch.flatten(features2, 1, 2)

        # Compute matching
        dists, idxs = KF.match_nn(descs1, descs2)
    
    return dists, img1, descs1, img2, descs2

'''Computing SIFT Descriptors on patches of the images before using the
nearest neighbour function for matching.'''
def kornia_matcher_test_descriptor_SIFTDescriptor(
        img1_name, img2_name, device, LocalFeature):
    # Load images from the given paths
    img1 = K.io.load_image(img1_name, K.io.ImageLoadType.GRAY32, 
                           device=device)[None, ...]        
    img2 = K.io.load_image(img2_name, K.io.ImageLoadType.GRAY32, 
                           device=device)[None, ...]

    # Cut images into patches for the descriptor.
    # Only works if img1 and img2 have the same dimensions
    patch_size = 32

    factor_x = int(img1.size(2) / patch_size)
    factor_y = int(img1.size(3) / patch_size)

    img1_p = []
    img2_p = []
    
    ## Cutting
    for x in range(factor_x):
        for y in range(factor_y):
            img1_p.append(
                    img1.data[:,:,x * patch_size:(x+1) * patch_size,
                              y * patch_size:(y+1) * patch_size]
            )
            img2_p.append(
                    img2.data[:,:,x * patch_size:(x+1) * patch_size,
                              y * patch_size:(y+1) * patch_size]
            )
    
    img1_p_tensor = torch.cat(img1_p, 0)
    img2_p_tensor = torch.cat(img2_p, 0)

    with torch.inference_mode():
        # Using SIFTDescriptor
        descs1 = LocalFeature(img1_p_tensor)
        descs2 = LocalFeature(img2_p_tensor)

        # Compute matching
        dists, idxs = KF.match_nn(descs1, descs2)
    
    return dists, img1, descs1, img2, descs2

''' The same as "kornia_matcher_test_descriptor_SIFTDescriptor" but 
with a patch size of only 8. Patch size could be turned in an argument,
making this function obsolete.'''
def kornia_matcher_test_descriptor_SIFTDescriptor_8(
        img1_name, img2_name, device, LocalFeature):
    # Load images from the given paths
    img1 = K.io.load_image(img1_name, K.io.ImageLoadType.GRAY32, 
                           device=device)[None, ...]        
    img2 = K.io.load_image(img2_name, K.io.ImageLoadType.GRAY32, 
                           device=device)[None, ...]

    # Cut images into patches for the descriptor.
    # Only works if img1 and img2 have the same dimensions
    patch_size = 8

    factor_x = int(img1.size(2) / patch_size)
    factor_y = int(img1.size(3) / patch_size)

    img1_p = []
    img2_p = []
    
    ## Cutting
    for x in range(factor_x):
        for y in range(factor_y):
            img1_p.append(
                    img1.data[:,:,x * patch_size:(x+1) * patch_size,
                              y * patch_size:(y+1) * patch_size]
            )
            img2_p.append(
                    img2.data[:,:,x * patch_size:(x+1) * patch_size,
                              y * patch_size:(y+1) * patch_size]
            )
    
    img1_p_tensor = torch.cat(img1_p, 0)
    img2_p_tensor = torch.cat(img2_p, 0)

    with torch.inference_mode():
        # Using SIFTDescriptor
        descs1 = LocalFeature(img1_p_tensor)
        descs2 = LocalFeature(img2_p_tensor)

        # Compute matching
        dists, idxs = KF.match_nn(descs1, descs2)
    
    return dists, img1, descs1, img2, descs2

'''Compute Multiple Kernel local descriptors of patches of the images 
before using the nearest neighbour function for matching.'''
def kornia_matcher_test_descriptor_MKDDescriptor(
        img1_name, img2_name, device, LocalFeature):
    # Load images from the given paths
    img1 = K.io.load_image(img1_name, K.io.ImageLoadType.GRAY32, 
                           device=device)[None, ...]        
    img2 = K.io.load_image(img2_name, K.io.ImageLoadType.GRAY32, 
                           device=device)[None, ...]

    # Cut images into patches for the descriptor.
    # Only works if img1 and img2 have the same dimensions
    patch_size = 32

    factor_x = int(img1.size(2) / patch_size)
    factor_y = int(img1.size(3) / patch_size)

    img1_p = []
    img2_p = []
    
    ## Cutting
    for x in range(factor_x):
        for y in range(factor_y):
            img1_p.append(
                    img1.data[:,:,x * patch_size:(x+1) * patch_size,
                              y * patch_size:(y+1) * patch_size]
            )
            img2_p.append(
                    img2.data[:,:,x * patch_size:(x+1) * patch_size,
                              y * patch_size:(y+1) * patch_size]
            )
    
    img1_p_tensor = torch.cat(img1_p, 0)
    img2_p_tensor = torch.cat(img2_p, 0)

    with torch.inference_mode():
        # Using MKDDescriptor
        descs1 = LocalFeature(img1_p_tensor)
        descs2 = LocalFeature(img2_p_tensor)

        # Compute matching
        dists, idxs = KF.match_nn(descs1, descs2)
    
    return dists, img1, descs1, img2, descs2

'''Computes Hardnet descriptors of patches of the images 
before using the nearest neighbour function for matching.'''
def kornia_matcher_test_descriptor_Hardnet(
        img1_name, img2_name, device, LocalFeature):
    # Load images from the given paths
    img1 = K.io.load_image(img1_name, K.io.ImageLoadType.GRAY32, 
                           device=device)[None, ...]        
    img2 = K.io.load_image(img2_name, K.io.ImageLoadType.GRAY32, 
                           device=device)[None, ...]

    # Cut images into patches for the descriptor.
    # Only works if img1 and img2 have the same dimensions
    patch_size = 32

    factor_x = int(img1.size(2) / patch_size)
    factor_y = int(img1.size(3) / patch_size)

    img1_p = []
    img2_p = []
    
    ## Cutting
    for x in range(factor_x):
        for y in range(factor_y):
            img1_p.append(
                    img1.data[:,:,x * patch_size:(x+1) * patch_size,
                              y * patch_size:(y+1) * patch_size]
            )
            img2_p.append(
                    img2.data[:,:,x * patch_size:(x+1) * patch_size,
                              y * patch_size:(y+1) * patch_size]
            )
    
    img1_p_tensor = torch.cat(img1_p, 0)
    img2_p_tensor = torch.cat(img2_p, 0)

    with torch.inference_mode():
        # Using Descriptor
        descs1 = LocalFeature(img1_p_tensor)
        descs2 = LocalFeature(img2_p_tensor)

        # Compute matching
        dists, idxs = KF.match_nn(descs1, descs2)
    
    return dists, img1, descs1, img2, descs2

'''Computes Hardnet8 descriptors of patches of the images 
before using the nearest neighbour function for matching. Only differs
from "kornia_matcher_test_descriptor_Hardnet" in the argument, so it is
technically also obsolete.'''
def kornia_matcher_test_descriptor_Hardnet8(
        img1_name, img2_name, device, LocalFeature):
    # Load images from the given paths
    img1 = K.io.load_image(img1_name, K.io.ImageLoadType.GRAY32, 
                           device=device)[None, ...]        
    img2 = K.io.load_image(img2_name, K.io.ImageLoadType.GRAY32, 
                           device=device)[None, ...]

    # Cut images into patches for the descriptor.
    # Only works if img1 and img2 have the same dimensions
    patch_size = 32

    factor_x = int(img1.size(2) / patch_size)
    factor_y = int(img1.size(3) / patch_size)

    img1_p = []
    img2_p = []
    
    ## Cutting
    for x in range(factor_x):
        for y in range(factor_y):
            img1_p.append(
                    img1.data[:,:,x * patch_size:(x+1) * patch_size,
                              y * patch_size:(y+1) * patch_size]
            )
            img2_p.append(
                    img2.data[:,:,x * patch_size:(x+1) * patch_size,
                              y * patch_size:(y+1) * patch_size]
            )
    
    img1_p_tensor = torch.cat(img1_p, 0)
    img2_p_tensor = torch.cat(img2_p, 0)

    with torch.inference_mode():
        # Using Descriptor
        descs1 = LocalFeature(img1_p_tensor)
        descs2 = LocalFeature(img2_p_tensor)

        # Compute matching
        dists, idxs = KF.match_nn(descs1, descs2)
    
    return dists, img1, descs1, img2, descs2

''' Computes HyNet descriptors of patches of the images 
before using the nearest neighbour function for matching.'''
def kornia_matcher_test_descriptor_HyNet(
        img1_name, img2_name, device, LocalFeature):
    # Load images from the given paths
    img1 = K.io.load_image(img1_name, K.io.ImageLoadType.GRAY32, 
                           device=device)[None, ...]        
    img2 = K.io.load_image(img2_name, K.io.ImageLoadType.GRAY32, 
                           device=device)[None, ...]

    # Cut images into patches for the descriptor.
    # Only works if img1 and img2 have the same dimensions
    patch_size = 32

    factor_x = int(img1.size(2) / patch_size)
    factor_y = int(img1.size(3) / patch_size)

    img1_p = []
    img2_p = []
    
    ## Cutting
    for x in range(factor_x):
        for y in range(factor_y):
            img1_p.append(
                    img1.data[:,:,x * patch_size:(x+1) * patch_size,
                              y * patch_size:(y+1) * patch_size]
            )
            img2_p.append(
                    img2.data[:,:,x * patch_size:(x+1) * patch_size,
                              y * patch_size:(y+1) * patch_size]
            )
    
    img1_p_tensor = torch.cat(img1_p, 0)
    img2_p_tensor = torch.cat(img2_p, 0)

    with torch.inference_mode():
        # Using Descriptor
        descs1 = LocalFeature(img1_p_tensor)
        descs2 = LocalFeature(img2_p_tensor)

        # Compute matching
        dists, idxs = KF.match_nn(descs1, descs2)
    
    return dists, img1, descs1, img2, descs2

'''Computes TFeat descriptors of patches of the images 
before using the nearest neighbour function for matching.'''
def kornia_matcher_test_descriptor_TFeat(
        img1_name, img2_name, device, LocalFeature):
    # Load images from the given paths
    img1 = K.io.load_image(img1_name, K.io.ImageLoadType.GRAY32, 
                           device=device)[None, ...]        
    img2 = K.io.load_image(img2_name, K.io.ImageLoadType.GRAY32, 
                           device=device)[None, ...]

    # Cut images into patches for the descriptor.
    # Only works if img1 and img2 have the same dimensions
    patch_size = 32

    factor_x = int(img1.size(2) / patch_size)
    factor_y = int(img1.size(3) / patch_size)

    img1_p = []
    img2_p = []
    
    ## Cutting
    for x in range(factor_x):
        for y in range(factor_y):
            img1_p.append(
                    img1.data[:,:,x * patch_size:(x+1) * patch_size,
                              y * patch_size:(y+1) * patch_size]
            )
            img2_p.append(
                    img2.data[:,:,x * patch_size:(x+1) * patch_size,
                              y * patch_size:(y+1) * patch_size]
            )
    
    img1_p_tensor = torch.cat(img1_p, 0)
    img2_p_tensor = torch.cat(img2_p, 0)

    with torch.inference_mode():
        descs1 = LocalFeature(img1_p_tensor)
        descs2 = LocalFeature(img2_p_tensor)

        # Compute matching
        dists, idxs = KF.match_nn(descs1, descs2)
    
    return dists, img1, descs1, img2, descs2

'''Computes SOSNet descriptors of patches of the images 
before using the nearest neighbour function for matching.'''
def kornia_matcher_test_descriptor_SOSNet(
        img1_name, img2_name, device, LocalFeature):
    # Load images from the given paths
    img1 = K.io.load_image(img1_name, K.io.ImageLoadType.GRAY32, 
                           device=device)[None, ...]        
    img2 = K.io.load_image(img2_name, K.io.ImageLoadType.GRAY32, 
                           device=device)[None, ...]

    # Cut images into patches for the descriptor.
    # Only works if img1 and img2 have the same dimensions
    patch_size = 32

    factor_x = int(img1.size(2) / patch_size)
    factor_y = int(img1.size(3) / patch_size)

    img1_p = []
    img2_p = []
    
    ## Cutting
    for x in range(factor_x):
        for y in range(factor_y):
            img1_p.append(
                    img1.data[:,:,x * patch_size:(x+1) * patch_size,
                              y * patch_size:(y+1) * patch_size]
            )
            img2_p.append(
                    img2.data[:,:,x * patch_size:(x+1) * patch_size,
                              y * patch_size:(y+1) * patch_size]
            )
    
    img1_p_tensor = torch.cat(img1_p, 0)
    img2_p_tensor = torch.cat(img2_p, 0)

    with torch.inference_mode():
        descs1 = LocalFeature(img1_p_tensor)
        descs2 = LocalFeature(img2_p_tensor)

        # Compute matching
        dists, idxs = KF.match_nn(descs1, descs2)
    
    return dists, img1, descs1, img2, descs2

#######################################################################
#
# END OF: Testing the usage of kornia Descriptors
#
# Testing the usage of kornia Descriptors and Detectors
#
#######################################################################

# Tutorial: https://kornia.github.io/tutorials/nbs/line_detection_and_matching_sold2.html
'''Detects and describes SOLD2 keypoints and matches them with the 
corresponding matching function.'''
def kornia_matcher_test_Desc_and_Dete_SOLD2_detector(
        img1_name, img2_name, device, feature_desc_and_dete):
    # Load images from the given paths
    img1 = K.io.load_image(img1_name, K.io.ImageLoadType.GRAY32, 
                           device=device)[None, ...]        
    img2 = K.io.load_image(img2_name, K.io.ImageLoadType.GRAY32, 
                           device=device)[None, ...]

    with torch.inference_mode():
        imgs = torch.cat([img1, img2], dim=0)

        # Use the feature_descriptor given through the arguments
        with torch.inference_mode():
            outputs = feature_desc_and_dete(imgs)
        
        line_seg1 = outputs["line_segments"][0]
        line_seg2 = outputs["line_segments"][1]
        desc1 = outputs["dense_desc"][0]
        desc2 = outputs["dense_desc"][1]

        # Compute matching
        with torch.inference_mode():
            matches = feature_desc_and_dete.match(
                line_seg1, line_seg2, desc1[None], desc2[None])
            valid_matches = matches != -1
            match_indices = matches[valid_matches]

            matched_lines1 = line_seg1[valid_matches]
            matched_lines2 = line_seg2[match_indices]
    
    return match_indices, img1, line_seg1, img2, line_seg2

# https://kornia.readthedocs.io/en/latest/feature.html#kornia.feature.DeDoDe
'''Detects and describes DeDoDe keypoints and matches them with the 
smnn function.'''
def kornia_matcher_test_Desc_and_Dete_DeDoDo(
        img1_name, img2_name, device, feature_desc_and_dete):
    # Load images from the given paths
    img1 = K.io.load_image(img1_name, K.io.ImageLoadType.RGB32, 
                           device=device)[None, ...]        
    img2 = K.io.load_image(img2_name, K.io.ImageLoadType.RGB32, 
                           device=device)[None, ...]

    with torch.inference_mode():
        imgs = torch.cat([img1, img2], dim=0)

        # Use the feature_descriptor given through the arguments
        keypoints, scores, features = feature_desc_and_dete(imgs)

        kps1, descs1 = features[0], features[0]
        kps2, descs2 = features[1], features[1]

        # Compute matching
        dists, idxs = KF.match_smnn(descs1, descs2, th=0.85)
    return dists, img1, descs1, img2, descs2

'''Detects and describes DISK keypoints and matches them with the 
smnn function.'''
def kornia_matcher_test_Desc_and_Dete_Disk(
        img1_name, img2_name, device, feature_desc_and_dete):
    # Load images from the given paths
    img1 = K.io.load_image(img1_name, K.io.ImageLoadType.RGB32, 
                           device=device)[None, ...]        
    img2 = K.io.load_image(img2_name, K.io.ImageLoadType.RGB32, 
                           device=device)[None, ...]

    with torch.inference_mode():
        inp = torch.cat([img1, img2], dim=0)

        # Use the feature_descriptor given through the arguments
        num_features = 128
        features1, features2 = feature_desc_and_dete(
            inp, num_features, pad_if_not_divisible=True)
        kps1, descs1 = features1.keypoints, features1.descriptors
        kps2, descs2 = features2.keypoints, features2.descriptors

        # Compute matching
        dists, idxs = KF.match_smnn(descs1, descs2, th=0.85) # 
    return dists, img1, descs1, img2, descs2

'''Detects and describes SIFT(-Feature) keypoints and matches them with the 
smnn function.'''
def kornia_matcher_test_Desc_and_Dete_SIFTFeature(
        img1_name, img2_name, device, feature_desc_and_dete):
    # Load images from the given paths
    img1 = K.io.load_image(img1_name, K.io.ImageLoadType.GRAY32, 
                           device=device)[None, ...]        
    img2 = K.io.load_image(img2_name, K.io.ImageLoadType.GRAY32, 
                           device=device)[None, ...]

    with torch.inference_mode():
        input = {"image0": img1, "image1": img2}

        matcher = kornia.feature.LocalFeatureMatcher(
            feature_desc_and_dete, 
            kornia.feature.DescriptorMatcher('smnn', 0.85))
        out = matcher(input)
        confidence = out['confidence']
        keypoints1 = out['keypoints0']
        keypoints2 = out['keypoints1']
    
    return confidence, img1, keypoints1, img2, keypoints2

'''Detects and describes keypoints with GFTTAffNetHardNet and matches 
them with the smnn function.'''
def kornia_matcher_test_Desc_and_Dete_GFTTAffNetHardNet(
        img1_name, img2_name, device, feature_desc_and_dete):
    # Load images from the given paths
    img1 = K.io.load_image(img1_name, K.io.ImageLoadType.GRAY32, 
                           device=device)[None, ...]        
    img2 = K.io.load_image(img2_name, K.io.ImageLoadType.GRAY32, 
                           device=device)[None, ...]

    with torch.inference_mode():
        input = {"image0": img1, "image1": img2}

        matcher = kornia.feature.LocalFeatureMatcher(
            feature_desc_and_dete, 
            kornia.feature.DescriptorMatcher('smnn', 0.85))
        out = matcher(input)
        confidence = out['confidence']
        keypoints1 = out['keypoints0']
        keypoints2 = out['keypoints1']
    
    return confidence, img1, keypoints1, img2, keypoints2

'''Detects and describes keypoints with KeyNetAffNetHardNet and matches 
them with the smnn function.'''
def kornia_matcher_test_Desc_and_Dete_KeyNetAffNetHardNet(
        img1_name, img2_name, device, feature_desc_and_dete):
    # Load images from the given paths
    img1 = K.io.load_image(img1_name, K.io.ImageLoadType.GRAY32, 
                           device=device)[None, ...]        
    img2 = K.io.load_image(img2_name, K.io.ImageLoadType.GRAY32, 
                           device=device)[None, ...]

    with torch.inference_mode():
        input = {"image0": img1, "image1": img2}

        matcher = kornia.feature.LocalFeatureMatcher(
            feature_desc_and_dete, 
            kornia.feature.DescriptorMatcher('smnn', 0.85))
        out = matcher(input)
        confidence = out['confidence']
        keypoints1 = out['keypoints0']
        keypoints2 = out['keypoints1']
    
    return confidence, img1, keypoints1, img2, keypoints2

#######################################################################
#
# END OF: Testing the usage of kornia Descriptors and Detectors
#
#######################################################################

# Influenced by: https://kornia.github.io/tutorials/nbs/descriptors_matching.html
'''Testing OpenCV SIFT with fginn matching.'''
def kornia_fginn_SIFT_test(img1_name, img2_name, device, 
                              feature_extractor):
    # Load images from the given paths
    img1 = cv2.cvtColor(cv2.imread(img1_name), cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(cv2.imread(img2_name), cv2.COLOR_BGR2RGB)

    with torch.inference_mode():

        # Use the feature_extractor, disk as the standard, to detect 
        # the features of the concatinated image
        sift = cv2.SIFT_create(8000)
        kps1, descs1 = sift.detectAndCompute(img1, None)
        kps2, descs2 = sift.detectAndCompute(img2, None)

        # Converting OPenCV SIFT keypoints to LAFs
        lafs1 = laf_from_opencv_SIFT_kpts(kps1)
        lafs2 = laf_from_opencv_SIFT_kpts(kps2)

        # Compute distances of the matching
        dists, idxs = KF.match_fginn(torch.from_numpy(descs1), 
                                     torch.from_numpy(descs2), 
                                     lafs1, lafs2, th=0.85)
        
        return dists, img1, lafs1, img2, lafs2

# Kornia also supports OpenCV [NOT WORKING YET]
def kornia_matching_OpenCV_test(img1_name, img2_name, device):
    img1 = cv2.imread(img1_name,0)          
    img2 = cv2.imread(img2_name,0)

    # OpenCV SIFT
    orb = cv2.ORB_create()
    kps1, descs1 = orb.detectAndCompute(img1, None)
    kps2, descs2 = orb.detectAndCompute(img2, None)

    with torch.inference_mode():
        descs1_tensor = torch.from_numpy(descs1).to(device=device).type(torch.float32)
        descs2_tensor = torch.from_numpy(descs2).to(device=device).type(torch.float32)
        
        dists, idxs = KF.match_smnn(descs1_tensor, descs2_tensor, 0.85)

    return dists, img1, descs1, img2, descs2

# https://kornia.github.io/tutorials/nbs/image_matching_lightglue.html
'''First tests with the kornia library.'''
def kornia_lightglue_matching(
        img1_name, img2_name, device, matcher, feature_extractor):
    img1 = K.io.load_image(img1_name, K.io.ImageLoadType.RGB32, 
                           device=device)[None, ...]        
    img2 = K.io.load_image(img2_name, K.io.ImageLoadType.RGB32, 
                           device=device)[None, ...]

    num_features = 64

    hw1 = torch.tensor(img1.shape[2:], device=device)
    hw2 = torch.tensor(img2.shape[2:], device=device)
    with torch.no_grad():
        inp = torch.cat([img1, img2], dim=0)
        features1, features2 = feature_extractor(
            inp, num_features, pad_if_not_divisible=True)
        kps1, descs1 = features1.keypoints, features1.descriptors
        kps2, descs2 = features2.keypoints, features2.descriptors
        lafs1 = KF.laf_from_center_scale_ori(
            kps1[None], torch.ones(1, len(kps1), 1, 1, device=device))
        lafs2 = KF.laf_from_center_scale_ori(
            kps2[None], torch.ones(1, len(kps2), 1, 1, device=device))
    dists, idxs = matcher(descs1, descs2, lafs1, lafs2, 
                          hw1=hw1, hw2=hw2)

    return dists, img1, kps1, img2, kps2, idxs

# https://kornia.github.io/tutorials/nbs/image_matching_lightglue.html
'''First tests with the kornia library.'''
def kornia_disk_lightglue_matching(
        img1_name, img2_name, device, matcher, feature_extractor):
    img1 = K.io.load_image(img1_name, K.io.ImageLoadType.RGB32, 
                           device=device)[None, ...]        
    img2 = K.io.load_image(img2_name, K.io.ImageLoadType.RGB32, 
                           device=device)[None, ...]

    num_features = 512

    hw1 = torch.tensor(img1.shape[2:], device=device)
    hw2 = torch.tensor(img2.shape[2:], device=device)
    with torch.inference_mode():
        inp = torch.cat([img1, img2], dim=0)
        features1, features2 = feature_extractor(
            inp, num_features, pad_if_not_divisible=True)
        kps1, descs1 = features1.keypoints, features1.descriptors
        kps2, descs2 = features2.keypoints, features2.descriptors
        lafs1 = KF.laf_from_center_scale_ori(
            kps1[None], torch.ones(1, len(kps1), 1, 1, device=device))
        lafs2 = KF.laf_from_center_scale_ori(
            kps2[None], torch.ones(1, len(kps2), 1, 1, device=device))
        dists, idxs = matcher(descs1, descs2, lafs1, lafs2, 
                              hw1=hw1, hw2=hw2)

    mkpts1, mkpts2 = get_matching_keypoints(kps1, kps2, idxs)

    # If the number of mkpts1 or mkpts2 gets to low(lower than 9?), 
    # this does not work. Higher num_features values fixed that.
    try:
        Fm, inliers = cv2.findFundamentalMat(
            mkpts1.detach().cpu().numpy(), 
            mkpts2.detach().cpu().numpy(), 
            cv2.USAC_ACCURATE, 1.0, 0.999, 100000)
    except:
        print(len(mkpts1), len(mkpts2))
        Fm, inliers = cv2.findFundamentalMat(
            mkpts1.detach().cpu().numpy(), 
            mkpts2.detach().cpu().numpy(), 
            cv2.USAC_ACCURATE, 1.0, 0.999, 100000)
    inliers = inliers > 0

    return inliers, img1, kps1, img2, kps2, idxs

''' CornerGTFF with SIFTDescriptor TEST'''
def kornia_detector_test_CornerGTFF_SIFT(
        img1_name, img2_name, device, feature_extractor):
    # Load images from the given paths
    img1 = K.io.load_image(img1_name, K.io.ImageLoadType.RGB32, 
                           device=device)[None, ...]        
    img2 = K.io.load_image(img2_name, K.io.ImageLoadType.RGB32, 
                           device=device)[None, ...]

    # Applying Detectors
    img1 = kornia.feature.dog_response_single(img1, sigma1=1.0, 
                                              sigma2=1.6)
    img1 = kornia.feature.dog_response_single(img2, sigma1=1.0, 
                                              sigma2=1.6)

    with torch.inference_mode():
        # Concatinate images
        inp = torch.cat([img1, img2], dim=0)

        # Use the feature_extractor, disk as the standard, to detect 
        # the features of the concatinated image
        num_features = 128
        features1, features2 = feature_extractor(
            inp, num_features, pad_if_not_divisible=True)
        kps1, descs1 = features1.keypoints, features1.descriptors
        kps2, descs2 = features2.keypoints, features2.descriptors

        # Compute matching
        dists, idxs = KF.match_smnn(descs1, descs2, th=0.85)
    
    return dists, img1, kps1, img2, kps2


#######################################################################
# Main function  ------------------------------------------------------
#######################################################################

'''This functions takes a directory to images and a method (from
MATCHING_METHODS) and computes the matching for the images with the
given method. If image_limit is a positive number, the number of images 
the matching is computed upon is reduced.'''
def extract_kornia_matches_in_directory(
        data_dir, method=1, image_limit=-1, print_log=False):
    
    device = K.utils.get_cuda_or_mps_device_if_available()
    print(device)

    matches = {}

    # Creates a MatchingHandler with the chosen method.
    # The MatchingHandler actually does all the matching with the 
    # images, while this function only is the frame
    mh = MatchingHandler(method=method)

    #create image dictionary
    counter_1 = 0
    for root1, dirs1, files1 in os.walk(data_dir, topdown=False):
        for name1 in files1:
            counter_1 += 1
            matches[name1] = {}
            if ((image_limit > 0) & (counter_1 > image_limit)):
                break
        if ((image_limit > 0) & (counter_1 > image_limit)):
                break

    # Extract matches. Counters here used for the log and image limit.
    counter_2 = 0
    for root1, dirs1, files1 in os.walk(data_dir, topdown=False):
        # Counter for progress printing
        if(image_limit > 0):
            counter_total = min(image_limit, len(files1))
        else:
            counter_total = len(files1)

        for name1 in files1:
            counter_2 += 1
            counter_3 = 0

            # Path for image one
            img1 = os.path.join(root1, name1)
            for root2, dirs2, files2 in os.walk(data_dir, topdown=False):
                for name2 in files2:
                    counter_3 += 1

                    # Path for image 2
                    img2 = os.path.join(root2, name2)

                    # Apply matching handler, receive similarity score
                    if img1 == img2: #Filter out pairs of the same image
                        matches[name1][name2] = \
                            mh.diagonal_forward(img1, img2)
                    else:
                        if name1 in matches:
                            if name2 in matches[name1]:
                                continue
                            matches[name1][name2] = matches[name2][name1] = mh.forward(img1, img2)                             
                    if(counter_3 >= counter_total):
                        break
                if(counter_3 >= counter_total):
                    break
            if print_log:
                print(counter_2, "/", counter_total)


            if(counter_2 >= counter_total):
                df = pd.DataFrame.from_dict(matches, orient="index")
                df.fillna(0, inplace=True)
                return df
            
    df = pd.DataFrame.from_dict(matches, orient="index")
    df.fillna(0, inplace=True) # Replace NA values with zeros
    return df

# Possible methods from the MatchingHandler
MATCHING_METHODS = [
    "ORB", # 0
    "matcher nn",
    "matcher mnn",
    "matcher snn",
    "matcher smnn",
    "matcher fginn",
    "matcher AdaLAM",
    "matcher lightglue",
    "matcher LoFTR",
    "detector gtff_response",
    "detector dog_response_single", # 10
    "kornia_descriptor_Dense_Sift_descriptor",
    "kornia_descriptor_SIFT_descriptor",
    "kornia_descriptor_MKDDescriptor",
    "kornia_descriptor_HardNet_descriptor",
    "kornia_descriptor_Hardnet8_descriptor",
    "kornia_descriptor_HyNet_descriptor",
    "kornia_descriptor_TFeat_descriptor",
    "kornia_descriptor_SOSNet_descriptor",
    "dete_and_dest SOLD2_detector",
    "dete_and_dest DeDoDe", # 20
    "dete_and_dest DISK",
    "dete_and_dest SIFTFeature",
    "dete_and_dest GFTTAffNetHardNet",
    "dete_and_dest KeyNetAffNetHardNet",
    "OpenCV 2nn",
    "Testing",
    "smnn_abs_count",
    "kornia nn ORB",
    "OpenCV smnn",
    "smnn DISK with estimator USAC_ACCURATE" # 30
]

'''A class, that makes usage of all the keypoint detection, description
and matching functions easy. It mainly contains three steps:
- Initialize creates instances of objects needed in the matching 
    process, so they get only created once.
- The forward method takes two images and computes the matching and 
    returnes the similarity score.
- The diagonal forward method returnes the standard value for matchings 
    of the same image. Appears to have mostly no influence.'''
class MatchingHandler():
    detector = None
    descriptor = None
    matcher = None
    method = 0

    # Initialize function, sets the method
    def __init__(self, method: int = 0):
        self.device = K.utils.get_cuda_or_mps_device_if_available()
        print("Matching Handler started with device " + str(self.device) + ".")
        self.switch_method(method)
    
    # Switches the method, calling the instancing of new classes
    def switch_method(self, method: int = 0) -> None:
        if self.method == method:
            return
        self.method = method
        self.initialize_method()
        print("Matching Handler now using method " + str(method) + ": " + str(MATCHING_METHODS[method]) + ".")

    # Resets the detector, descriptor and matcher and loads in new ones,
    # Depending on the method.
    def initialize_method(self) -> None:
        # Reset computation function values for a cleaner algorithm
        self.detector, self.descriptor, self.matcher = None, None, None
        
        match self.method:
            case 0:
                # Original ORB method
                pass
            case 1:
                # kornia_matcher_test_nn
                self.detector = KF.DISK.from_pretrained("depth").to(self.device)
            case 2:
                # kornia_matcher_test_mnn
                self.detector = KF.DISK.from_pretrained("depth").to(self.device)
            case 3:
                # kornia_matcher_test_snn
                self.detector = KF.DISK.from_pretrained("depth").to(self.device)
            case 4:
                # kornia_matcher_test_smnn
                self.detector = KF.DISK.from_pretrained("depth").to(self.device)
            case 5:
                # kornia_matcher_test_fginn
                self.detector = KF.DISK.from_pretrained("depth").to(self.device)
            case 6:
                # kornia_matcher_test_adalam
                self.detector = KF.DISK.from_pretrained("depth").to(self.device)
            case 7:
                # kornia_matcher_test_lightglue
                self.detector = KF.DISK.from_pretrained("depth").to(self.device)
                self.matcher = KF.LightGlueMatcher("disk").eval().to(self.device)
            case 8:
                # kornia_matcher_test_LoFTR
                self.matcher = KF.LoFTR(pretrained="indoor_new").to(self.device)
            case 9:
                # kornia_detector_test_gftt_response
                self.detector = KF.DISK.from_pretrained("depth").to(self.device)
            case 10:
                # kornia_detector_test_dog_response_single
                self.detector = KF.DISK.from_pretrained("depth").to(self.device)
            case 11:
                # kornia_descriptor_Dense_Sift_descriptor
                self.descriptor = kornia.feature.DenseSIFTDescriptor().to(self.device)
            case 12:
                # kornia_descriptor_SIFT_descriptor
                self.descriptor = kornia.feature.SIFTDescriptor(32, 8, 4).to(self.device)
            case 13:
                # kornia_descriptor_MKDDescriptor
                self.descriptor = kornia.feature.MKDDescriptor().to(self.device)
            case 14:
                # kornia_descriptor_HardNet_descriptor
                self.descriptor = kornia.feature.HardNet(pretrained=True).to(self.device)
            case 15:
                # kornia_descriptor_Hardnet8_descriptor
                self.descriptor = kornia.feature.HardNet8(pretrained=True).to(self.device)
            case 16:
                # kornia_descriptor_HyNet_descriptor
                self.descriptor = kornia.feature.HyNet(pretrained=True).to(self.device)
            case 17:
                # kornia_descriptor_TFeat_descriptor
                self.descriptor = kornia.feature.TFeat(pretrained=True).to(self.device)
            case 18:
                # kornia_descriptor_SOSNet_descriptor
                self.descriptor = kornia.feature.SOSNet(pretrained=True).to(self.device)
            case 19:
                # kornia_test_Desc_and_Dete_SOLD2_detector
                new_cfg: Dict[str, Any] = {
                    "backbone_cfg": {"input_channel": 1, "depth": 4, "num_stacks": 2, "num_blocks": 1, "num_classes": 5},
                    "use_descriptor": True,
                    "grid_size": 4, # [8]
                    "keep_border_valid": True,
                    "detection_thresh": 0.0153846,  # = 1/65: threshold of junction detection
                    "max_num_junctions": 125,  # [500] maximum number of junctions per image
                    "line_detector_cfg": {
                        "detect_thresh": 0.5,
                        "num_samples": 32, # [64]
                        "inlier_thresh": 0.99,
                        "use_candidate_suppression": True,
                        "nms_dist_tolerance": 2.0, # [3.0]
                        "use_heatmap_refinement": True,
                        "heatmap_refine_cfg": {
                            "mode": "local",
                            "ratio": 0.2,
                            "valid_thresh": 0.001,
                            "num_blocks": 10, # [20]
                            "overlap_ratio": 0.5,
                        },
                        "use_junction_refinement": True,
                        "junction_refine_cfg": {"num_perturbs": 9, "perturb_interval": 0.25},
                    },
                    "line_matcher_cfg": {
                        "cross_check": True,
                        "num_samples": 3, # [5]
                        "min_dist_pts": 4, # [8]
                        "top_k_candidates": 5, # [10]
                        "grid_size": 2, # [4]
                    },
                }
                self.descriptor = kornia.feature.SOLD2(pretrained=True, config=new_cfg).to(self.device)  
            case 20:
                # kornia_test_Desc_and_Dete_DeDoDo
                self.descriptor = kornia.feature.DeDoDe().from_pretrained().to(self.device)
            case 21:
                # kornia_test_Desc_and_Dete_Disk
                self.descriptor = kornia.feature.DISK.from_pretrained('depth', device=self.device)
            case 22:
                # kornia_test_Desc_and_Dete_SIFTFeature
                self.descriptor = kornia.feature.SIFTFeature(device=self.device)
            case 23:
                # kornia_test_Desc_and_Dete_GFTTAffNetHardNet
                self.descriptor = KF.GFTTAffNetHardNet(device=self.device)
            case 24:
                # kornia_test_Desc_and_Dete_KeyNetHardNet
                self.descriptor = KF.KeyNetHardNet(device=self.device)
            case 25:
                # OpenCV knn method with k=2
                pass
            case 26:
                # Test
                #detector = KF.BlobDoGSingle()
                #descriptor = KF.LAFDescriptor(KF.HardNet(pretrained=True))
                #self.detector = KF.LocalFeature(detector, descriptor)
                self.descriptor = self.descriptor = kornia.feature.SIFTDescriptor(8, 8, 4).to(self.device)
            case 27:
                # smnn abs count (th = 0.85)
                self.detector = KF.DISK.from_pretrained("depth").to(self.device)
            case 28:
                # ORB with kornia nn matching
                pass
            case 29:
                # smnn in OpenCV
                pass
            case 30:
                # estimator test (else like 4)
                self.detector = KF.DISK.from_pretrained("depth").to(self.device)
            case _:
                print("Method " + str(self.method) + " does not exists")

    # Computes a matching of two images (given from paths) and returns
    # the similarity score
    def forward(self, image_1: str, image_2: str) -> float:
        match self.method:
            case 0:
                # For testing purposes the detect_keypoints_and_match function would temporarily replaces by other functions
                matches_found = detect_keypoints_and_match(image_1, image_2)

                good_matches = []
                for m in matches_found[0]:
                    if m.distance <= 45: # standard 45
                        good_matches.append(m)
                return len(good_matches)
            case 1:
                # kornia_matcher_test_nn
                matches_found = kornia_matcher_test_nn(image_1, image_2, device=self.device, feature_extractor=self.detector)
                x = matches_found[0]
                return x.sum().item()
            case 2:
                # kornia_matcher_test_mnn
                matches_found = kornia_matcher_test_mnn(image_1, image_2, device=self.device, feature_extractor=self.detector)
                x = matches_found[0]
                return torch.sum(torch.where(x < 1, 1, 0)).cpu().detach().numpy()
            case 3:
                # kornia_matcher_test_snn (th = 0.85)
                matches_found = kornia_matcher_test_snn(image_1, image_2, device=self.device, feature_extractor=self.detector)
                x = matches_found[0]
                return x.sum().item()
            case 4:
                # kornia_matcher_test_smnn (th = 0.85)
                matches_found = kornia_matcher_test_smnn(image_1, image_2, device=self.device, feature_extractor=self.detector)
                x = matches_found[0]
                #return x.sum().item()
                return x.sum().item()
            case 5:
                # kornia_matcher_test_fginn (th = 0.85)
                matches_found = kornia_matcher_test_fginn(image_1, image_2, device=self.device, feature_extractor=self.detector)
                x = matches_found[0]
                return len(x)
            case 6:
                # kornia_matcher_test_adalam
                matches_found = kornia_matcher_test_adalam(image_1, image_2, device=self.device, feature_extractor=self.detector)
                x = matches_found[0]
                return x.sum().item()
            case 7:
                # kornia_matcher_test_lightglue
                matches_found = kornia_matcher_test_lightglue(image_1, image_2, device=self.device, feature_extractor=self.detector, matcher=self.matcher)
                x = matches_found[0]
                return x.sum().cpu().detach().numpy()
            case 8:
                # kornia_matcher_test_LoFTR
                matches_found = kornia_matcher_test_LoFTR(image_1, image_2, device=self.device, matcher=self.matcher)
                x = matches_found[0]
                return x.sum().item()
            case 9:
                # kornia_detector_test_gftt_response
                matches_found = kornia_detector_test_gftt_response(image_1, image_2, device=self.device, feature_extractor=self.detector)
                x = matches_found[0]
                return x.sum().item()
            case 10:
                # kornia_detector_test_dog_response_single
                matches_found = kornia_detector_test_dog_response_single(image_1, image_2, device=self.device, feature_extractor=self.detector)
                x = matches_found[0]
                return x.sum().item()
            case 11:
                # kornia_descriptor_Dense_Sift_descriptor
                matches_found = kornia_matcher_test_descriptor_DenseSIFTDescriptor(image_1, image_2, device=self.device, feature_descriptor=self.descriptor)

                x = matches_found[0]
                return x.sum().item()
            case 12:
                # kornia_descriptor_SIFT_descriptor
                matches_found = kornia_matcher_test_descriptor_SIFTDescriptor(image_1, image_2, device=self.device, LocalFeature=self.descriptor)

                x = matches_found[0]
                return x.sum().item()
            case 13:
                # kornia_descriptor_MKDDescriptor
                matches_found = kornia_matcher_test_descriptor_MKDDescriptor(image_1, image_2, device=self.device, LocalFeature=self.descriptor)

                x = matches_found[0]
                return x.sum().item()
            case 14:
                # kornia_descriptor_HardNet_descriptor
                matches_found = kornia_matcher_test_descriptor_Hardnet(image_1, image_2, device=self.device, LocalFeature=self.descriptor)

                x = matches_found[0]
                return x.sum().item()
            case 15:
                # kornia_descriptor_Hardnet8_descriptor
                matches_found = kornia_matcher_test_descriptor_Hardnet8(image_1, image_2, device=self.device, LocalFeature=self.descriptor)

                x = matches_found[0]
                return x.sum().item()
            case 16:
                # kornia_descriptor_HyNet_descriptor
                matches_found = kornia_matcher_test_descriptor_HyNet(image_1, image_2, device=self.device, LocalFeature=self.descriptor)

                x = matches_found[0]
                return x.sum().item()
            case 17:
                # kornia_descriptor_TFeat_descriptor
                matches_found = kornia_matcher_test_descriptor_TFeat(image_1, image_2, device=self.device, LocalFeature=self.descriptor)

                x = matches_found[0]
                return x.sum().item()
            case 18:
                # kornia_descriptor_SOSNet_descriptor
                matches_found = kornia_matcher_test_descriptor_SOSNet(image_1, image_2, device=self.device, LocalFeature=self.descriptor)

                x = matches_found[0]
                return x.sum().item()
            case 19:
                # kornia_test_Desc_and_Dete_SOLD2_detector
                matches_found = kornia_matcher_test_Desc_and_Dete_SOLD2_detector(image_1, image_2, device=self.device, feature_desc_and_dete=self.descriptor)

                return torch.sum(matches_found[0]).cpu().detach().numpy()
            case 20:
                # kornia_test_Desc_and_Dete_DeDoDo
                matches_found = kornia_matcher_test_Desc_and_Dete_DeDoDo(image_1, image_2, device=self.device, feature_desc_and_dete=self.descriptor)

                return torch.sum(matches_found[0]).cpu().detach().numpy()
            case 21:
                # kornia_test_Desc_and_Dete_Disk
                matches_found = kornia_matcher_test_Desc_and_Dete_Disk(image_1, image_2, device=self.device, feature_desc_and_dete=self.descriptor)

                return torch.sum(matches_found[0]).cpu().detach().numpy()
            case 22:
                # kornia_test_Desc_and_Dete_SIFTFeature
                matches_found = kornia_matcher_test_Desc_and_Dete_SIFTFeature(image_1, image_2, device=self.device, feature_desc_and_dete=self.descriptor)

                return torch.sum(matches_found[0]).cpu().detach().numpy()
            case 23:
                # kornia_test_Desc_and_Dete_GFTTAffNetHardNet
                matches_found = kornia_matcher_test_Desc_and_Dete_GFTTAffNetHardNet(image_1, image_2, device=self.device, feature_desc_and_dete=self.descriptor)

                return torch.sum(matches_found[0]).cpu().detach().numpy()
            case 24:
                # kornia_test_Desc_and_Dete_KeyNetHardNet
                matches_found = kornia_matcher_test_Desc_and_Dete_KeyNetAffNetHardNet(image_1, image_2, device=self.device, feature_desc_and_dete=self.descriptor)

                return torch.sum(matches_found[0]).cpu().detach().numpy()
            case 25:
                # OpenCV knn method with k=2
                matches_found = detect_keypoints_and_descriptors_knn_match(image_1, image_2)

                good_matches = []
                for m in matches_found[0]:
                    if m.distance <= 45:
                        good_matches.append(m)
                return len(good_matches)
            case 26:
                #matches_found = descriptor_test(image_1, image_2, device=self.device, LocalFeature=self.detector)
                matches_found = kornia_matcher_test_descriptor_SIFTDescriptor_8(image_1, image_2, device=self.device, LocalFeature=self.descriptor)
                x = matches_found[0]
                return x.sum().item()
            case 27:
                # smnn abs count (th = 0.85)
                matches_found = kornia_matcher_test_smnn(image_1, image_2, device=self.device, feature_extractor=self.detector, th=0.85)
                x = matches_found[0]
                return x.size(dim=0)
            case 28:
                 # For testing purposes the detect_keypoints_and_match function would temporarily replaces by other functions
                matches_found = kornia_matching_OpenCV_test(image_1, image_2, device=self.device)

                good_matches = []
                for m in matches_found[0]:
                    if m <= 350: # standard 45
                        good_matches.append(m)
                return len(good_matches)
            case 29:
                # smnn in OpenCV
                # DOES NOT WORK, BECAUSE CROSS_CHECK AND RATIO TEST ARE
                # MUTUALLY EXCLUSIVE FOR THE BFMatcher
                matches_found = detect_keypoints_and_descriptors_knn_match(image_1, image_2, ch=True)

                good_matches = []
                for m in matches_found[0]:
                    if m.distance <= 45: # standard 45
                        good_matches.append(m)
                return len(good_matches)
            case 30:
                # estimator test (else like 4)
                matches_found = kornia_matcher_test_smnn_estimator(image_1, image_2, device=self.device, feature_extractor=self.detector)
                x = matches_found[0]
                #return x.sum().item()
                return x.sum().item()
            case _:
                pass
        return None
    
    # Returns a standard value for the matching
    def diagonal_forward(self, image_1: str, image_2: str) -> float:
        match self.method:
            case 0:
                return 0
            case 1:
                # kornia_matcher_test_nn
                return 0
            case 2:
                # kornia_matcher_test_mnn
                return 0
            case 3:
                # kornia_matcher_test_snn (th = 0.85)
                return 0
            case 4:
                # kornia_matcher_test_smnn (th = 0.85)
                return 0
            case 5:
                # kornia_matcher_test_fginn (th = 0.85)
                return 128
            case 6:
                # kornia_matcher_test_adalam
                return 0
            case 7:
                # kornia_matcher_test_lightglue
                return 0
            case 8:
                # kornia_matcher_test_LoFTR
                return 0
            case 9:
                # kornia_detector_test_gftt_response
                return 0
            case 10:
                # kornia_detector_test_dog_response_single
                return 0
            case 11:
                # kornia_descriptor_Dense_Sift_descriptor
                return 0
            case 12:
                # kornia_descriptor_SIFT_descriptor
                return 0
            case 13:
                # kornia_descriptor_MKDDescriptor
                return 0
            case 14:
                # kornia_descriptor_HardNet_descriptor
                return 0
            case 15:
                # kornia_descriptor_Hardnet8_descriptor
                return 0
            case 16:
                # kornia_descriptor_HyNet_descriptor
                return 0
            case 17:
                # kornia_descriptor_TFeat_descriptor
                return 0
            case 18:
                # kornia_descriptor_SOSNet_descriptor
                return 0
            case 19:
                # kornia_test_Desc_and_Dete_SOLD2_detector
                return 0
            case 20:
                # kornia_test_Desc_and_Dete_DeDoDo
                return 0
            case 21:
                # kornia_test_Desc_and_Dete_Disk
                return 0
            case 22:
                # kornia_test_Desc_and_Dete_SIFTFeature
                return 0
            case 23:
                # kornia_test_Desc_and_Dete_GFTTAffNetHardNet
                return 0
            case 24:
                # kornia_test_Desc_and_Dete_KeyNetHardNet
                return 0
            case 25:
                # OpenCV knn method with k=2
                return 0
            case 26:
                # test
                return 0
            case 26:
                # smnn abs count (th = 0.85)
                return 0
            case 27:
                return 0
            case 28:
                return 0
            case 29:
                # smnn in OpenCV
                return 0
            case 30:
                # estimator test (else like 4)
                return 0
            case _:
                pass
        return None
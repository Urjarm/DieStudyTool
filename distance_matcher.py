''' File based on utils.py from https://github.com/Frankfurt-BigDataLab/2023_CAA_ClaReNet/tree/main/Die_Study'''
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''The original matching function with ORB based on the implementation
by Deligio et al.. The options of BRIEF and BRISK for keypoint 
detection and description were removed, adaptability of cross_check
was added.'''
def detect_keypoints_and_match(img1_name, img2_name, cc=False):
    img1 = cv2.imread(img1_name,0)          
    img2 = cv2.imread(img2_name,0)

    finder = cv2.ORB_create()
    kp1, des1 = finder.detectAndCompute(img1,None)
    kp2, des2 = finder.detectAndCompute(img2,None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING2, crossCheck=cc)
    matches = bf.match(des1,des2)
    matches = sorted(matches, key = lambda x:x.distance)

    return matches, img1, kp1, img2, kp2

'''Using SIFT instead of ORB with a different distance metric for
matching that is recommended for SIFT.'''
def detect_keypoints_and_match_SIFT(img1_name, img2_name):
    img1 = cv2.imread(img1_name,0)          
    img2 = cv2.imread(img2_name,0)

    finder = cv2.SIFT_create()
    kp1, des1 = finder.detectAndCompute(img1,None)
    kp2, des2 = finder.detectAndCompute(img2,None)

    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.match(des1,des2)
    matches = sorted(matches, key = lambda x:x.distance)

    return matches, img1, kp1, img2, kp2

'''Using BRISK instead of ORB with a different distance metric for
matching.'''
def detect_keypoints_and_match_BRISK(img1_name, img2_name):
    img1 = cv2.imread(img1_name,0)          
    img2 = cv2.imread(img2_name,0)

    finder = cv2.BRISK_create()
    kp1, des1 = finder.detectAndCompute(img1,None)
    kp2, des2 = finder.detectAndCompute(img2,None)

    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.match(des1,des2)
    matches = sorted(matches, key = lambda x:x.distance)

    return matches, img1, kp1, img2, kp2

'''Using Lowe's ratio test after matching keypoints from ORB.'''
def detect_keypoints_and_descriptors_knn_match(img1_name, img2_name, 
                                               ratio_thresh = 0.75, 
                                               ch=False):
    img1 = cv2.imread(img1_name,0)          
    img2 = cv2.imread(img2_name,0)

    finder = cv2.ORB_create()
    kp1, des1 = finder.detectAndCompute(img1,None)
    kp2, des2 = finder.detectAndCompute(img2,None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING2, crossCheck=ch)
    knn_matches = bf.knnMatch(des1, des2, 2)
    good_matches = []
    for m,n in knn_matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)

    return good_matches, img1, kp1, img2, kp2, des1, des2

'''Testing a different kind of matcher.'''
def detect_keypoints_match_hamming(img1_name, img2_name):
    img1 = cv2.imread(img1_name,0)          
    img2 = cv2.imread(img2_name,0)

    finder = cv2.ORB_create()
    kp1, des1 = finder.detectAndCompute(img1,None)
    kp2, des2 = finder.detectAndCompute(img2,None)

    matcher = cv2.DescriptorMatcher_create(
        cv2.DescriptorMatcher_BRUTEFORCE_HAMMING)
    matches = matcher.match(des1,des2)
    matches = sorted(matches, key = lambda x:x.distance)

    return matches, img1, kp1, img2, kp2, des1, des2
    
'''Testing the FLANN matcher.'''
# https://www.tutorialspoint.com/how-to-implement-flann-based-feature-matching-in-opencv-python & https://docs.opencv.org/3.4/d5/d6f/tutorial_feature_flann_matcher.html
def flann_matcher(img1_name, img2_name):
    img1 = cv2.imread(img1_name,0)          
    img2 = cv2.imread(img2_name,0)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params,search_params)
    knn_matches  = flann.knnMatch(des1,des2,k=2)
    
    #-- Filter matches using the Lowe's ratio test
    # ratio_thresh = 0.7
    # good_matches = []
    # for m,n in knn_matches :
    #     if m.distance < ratio_thresh * n.distance:
    #         good_matches.append(m)

    return knn_matches, img1, kp1, img2, kp2

def visualise_matches(img1, kp1, img2, kp2, matches):
    img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches,None, flags=2)
    plt.figure(figsize=(15,10))
    plt.imshow(img3)
    plt.show()

def extract_matches_in_directory(data_dir, method=1, max_distance=45):
    """
    Applys detect_keypoints and matches to all elements in a directory. Please make sure that there are only images in the directory,
    else also some exceptions can be added.

    count defines if the sum of matches should be returned or the list of found matches.
    """
    matches = {}

    #create image dictionary
    for root1, dirs1, files1 in os.walk(data_dir, topdown=False):
        for name1 in files1:
            matches[name1] = {}

    # extract matches 
    for root1, dirs1, files1 in os.walk(data_dir, topdown=False):
        for name1 in files1:
            img1 = os.path.join(root1, name1)

            for root2, dirs2, files2 in os.walk(data_dir, topdown=False):
                for name2 in files2:
                    img2 = os.path.join(root2, name2)

                    if name1 == name2:
                        matches[name1][name2] = 0 # To avoid wrong orderings in columns

                    else:
                        if name1 in matches:
                            if name2 in matches[name1]:
                                continue
                            else:

                                match method:
                                    case 1:
                                        # abs
                                        matches_found = detect_keypoints_and_match(img1, img2)

                                        good_matches = []
                                        for m in matches_found[0]:
                                            if m.distance <= max_distance:
                                                good_matches.append(m)
                                        matches[name1][name2] = len(good_matches)
                                        matches[name2][name1] = len(good_matches)
                                    case 2:
                                        # norm dist

                                        # There are m matches and the maximum distance n can be computed with pythagoras.
                                        # => Therefore the product of these two acts as a normalizer here.
                                        matches_found = detect_keypoints_and_match(img1, img2)

                                        normalizer = len(matches_found[0]) # * np.sqrt(224**2 + 224**2)  # READ IN IMAGE-SIZES FROM THE IMAGE INSTEAD OF 224?
                                        distance = 0

                                        for m in matches_found[0]:
                                            distance += m.distance / normalizer

                                        if distance == 0:
                                            matches[name1][name2] = distance
                                            matches[name2][name1] = distance
                                        else:
                                            matches[name1][name2] = 1/distance
                                            matches[name2][name1] = 1/distance
                                    case 3:
                                        # norm dist times amount of good matches
                                        matches_found = detect_keypoints_and_match(img1, img2)

                                        good_matches = []
                                        for m in matches_found[0]:
                                            if m.distance <= max_distance:
                                                good_matches.append(m)

                                        normalizer = len(matches_found[0])
                                        distance = 0

                                        for m in matches_found[0]:
                                            distance += m.distance / normalizer

                                        if distance == 0:
                                            matches[name1][name2] = 1
                                            matches[name2][name1] = 1
                                        else:
                                            matches[name1][name2] = 1/distance * len(good_matches)
                                            matches[name2][name1] = 1/distance * len(good_matches)
                                        # ---------------------
                                    case 4:
                                        # Abs FLANN matcher
                                        matches_found = flann_matcher(img1, img2)

                                        good_matches = []
                                        for m in matches_found[0]:
                                            if m.distance <= max_distance:
                                                good_matches.append(m)
                                        matches[name1][name2] = len(good_matches)
                                        matches[name2][name1] = len(good_matches)
                                    case 5:
                                        # Hamming matcher
                                        matches_found = detect_keypoints_match_hamming(img1, img2)

                                        good_matches = []
                                        for m in matches_found[0]:
                                            if m.distance <= max_distance:
                                                good_matches.append(m)
                                        matches[name1][name2] = len(good_matches)
                                        matches[name2][name1] = len(good_matches)
                                    case 6:
                                        # knn matcher (2)
                                        matches_found = detect_keypoints_and_descriptors_knn_match(img1, img2)

                                        good_matches = []
                                        for m in matches_found[0]:
                                            if m.distance <= max_distance:
                                                good_matches.append(m)
                                        matches[name1][name2] = len(good_matches)
                                        matches[name2][name1] = len(good_matches)
                                    case _:
                                        print("Method " + str(method) + " does not exists")
                                        matches = []
    df = pd.DataFrame.from_dict(matches, orient="index")
    df.fillna(0, inplace=True)
    return df

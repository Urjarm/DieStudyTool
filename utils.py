''' File based on utils.py from https://github.com/Frankfurt-BigDataLab/2023_CAA_ClaReNet/tree/main/Die_Study'''
import os
import numpy as np
import cv2
import pandas as pd
import scipy.spatial
import scipy.special
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics as sk_metrics
import scipy
from sklearn.preprocessing import normalize
from rembg import remove
import skimage as skimage
from skimage.restoration import denoise_tv_chambolle
from skimage import exposure
import sys as sys

from sknetwork.clustering import PropagationClustering
from umap import UMAP
from sklearn.metrics import silhouette_score
import tqdm

import Orange

def apply_circle_crop(src, img_size=224, percentage=0.95):
	image = cv2.imread(src)
	image = cv2.resize(image, (img_size, img_size)) 
	circle_img = np.zeros((img_size, img_size), np.uint8)
	cv2.circle(circle_img, ((int)(img_size/2),(int)(img_size/2)), 
			int(img_size/2*percentage), 1, thickness=-1)
	image = cv2.bitwise_and(image, image, mask=circle_img)

	return image

def circle_crop_directory(src, target, img_size=224, percentage=0.95):
	
	for root, dirs, files in os.walk(src, topdown=False):
		for name in files:
			img = os.path.join(root, name)
			image = apply_circle_crop(img, img_size, percentage)
			cv2.imwrite(os.path.join(target, name), image)


def apply_grayscale(src, img_size=224):
	image = cv2.imread(src)
	try:
		image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	except:
		print(src)
	image = cv2.resize(image, (img_size, img_size), 
					interpolation = cv2.INTER_LINEAR)

	return image

def grayscale_directory(src, target, img_size=224):
	
		for root, dirs, files in os.walk(src, topdown=False):
			for name in files:
				img = os.path.join(root, name)
				image = apply_grayscale(img, img_size)
				cv2.imwrite(os.path.join(target, name), image)


def apply_clahe(src):

    image = cv2.imread(src)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(2, 2))
    equalized = clahe.apply(gray)

    return equalized

def clahe_directory(src, target, img_size=224):

	for root, dirs, files in os.walk(src, topdown=False):
		for name in files:
			img = os.path.join(root, name)
			image = apply_clahe(img)
			cv2.imwrite(os.path.join(target, name), image)

def remove_background(src, image_return_size=224):
	image = cv2.imread(src)
	image = cv2.resize(image, (image_return_size, image_return_size)) 
	image = remove(image, alpha_matting_background_threshold=10,
				alpha_matting_foreground_threshold=240)

	return image

def remove_background_directory(src, target, img_size=224):

	for root, dirs, files in os.walk(src, topdown=False):
		for name in files:
			img = os.path.join(root, name)
			image = remove_background(img, img_size)
			cv2.imwrite(os.path.join(target, name), image)

def apply_laplace(src, image_return_size=224):
	image = cv2.imread(src)
	image = cv2.resize(image, (image_return_size, image_return_size)) 
	image = cv2.Laplacian(image, cv2.CV_16S, None, 9)
	return image

def apply_laplace_directory(src, target, img_size=224):
	for root, dirs, files in os.walk(src, topdown=False):
		for name in files:
			img = os.path.join(root, name)
			image = apply_laplace(img, img_size)
			cv2.imwrite(os.path.join(target, name), image)

def apply_opencv_denoise(src, image_return_size=224, h=15):
	image = cv2.imread(src)
	image = cv2.resize(image, (image_return_size, image_return_size)) 
	image = cv2.fastNlMeansDenoising(image, None, h, templateWindowSize=9, searchWindowSize=27)
	return image

def apply_opencv_denoise_directory(src, target, img_size=224, h=15):
	for root, dirs, files in os.walk(src, topdown=False):
		for name in files:
			img = os.path.join(root, name)
			image = apply_opencv_denoise(img, img_size, h)
			cv2.imwrite(os.path.join(target, name), image)

def apply_median_blur(src, image_return_size=224, h=5):
	image = cv2.imread(src)
	image = cv2.resize(image, (image_return_size, image_return_size)) 
	image = cv2.medianBlur(image, h)
	return image

def apply_median_blur_directory(src, target, img_size=224, h=5):
	for root, dirs, files in os.walk(src, topdown=False):
		for name in files:
			img = os.path.join(root, name)
			image = apply_median_blur(img, img_size, h)
			cv2.imwrite(os.path.join(target, name), image)


def apply_gaussian_blur(src, image_return_size=224, h=5):
	image = cv2.imread(src)
	image = cv2.resize(image, (image_return_size, image_return_size)) 
	image = cv2.GaussianBlur(image, (h, h), 2)
	return image

def apply_gaussian_blur_directory(src, target, img_size=224, h=5):
	for root, dirs, files in os.walk(src, topdown=False):
		for name in files:
			img = os.path.join(root, name)
			image = apply_gaussian_blur(img, img_size, h)
			cv2.imwrite(os.path.join(target, name), image)


def apply_bilateral_blur(src, image_return_size=224, h=5, stre=150):
	image = cv2.imread(src)
	image = cv2.resize(image, (image_return_size, image_return_size)) 
	image = cv2.bilateralFilter(image, h, stre, 10)
	return image
			
def apply_bilateral_blur_directory(src, target, img_size=224, h=5, stre=150):
	for root, dirs, files in os.walk(src, topdown=False):
		for name in files:
			img = os.path.join(root, name)
			image = apply_bilateral_blur(img, img_size, h, stre)
			cv2.imwrite(os.path.join(target, name), image)


def apply_median_blur_no_resize(src, h=9):
	image = cv2.imread(src)
	image = cv2.medianBlur(image, h)
	return image

def apply_median_blur_no_resize_directory(src, target, h=9):
	for root, dirs, files in os.walk(src, topdown=False):
		for name in files:
			img = os.path.join(root, name)
			image = apply_median_blur_no_resize(img, h)
			cv2.imwrite(os.path.join(target, name), image)


def apply_histogram_equalization(src, image_return_size=224):
	image = cv2.imread(src)
	image = cv2.resize(image, (image_return_size, image_return_size))
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	image = cv2.equalizeHist(image, None)
	return image

def histogram_equalization_directory(src, target, img_size=224):
	for root, dirs, files in os.walk(src, topdown=False):
		for name in files:
			img = os.path.join(root, name)
			image = apply_histogram_equalization(img, img_size)
			cv2.imwrite(os.path.join(target, name), image)


def adaptive_threshhold_segmentation(src, image_return_size=224):
	image = cv2.imread(src)
	image = cv2.resize(image, (image_return_size, image_return_size))
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	image = cv2.adaptiveThreshold(image, 255, 
							   cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
							   cv2.THRESH_BINARY_INV, 11, 1)
	return image

def adaptive_threshhold_segmentation_directory(src, target, img_size=224):
	for root, dirs, files in os.walk(src, topdown=False):
		for name in files:
			img = os.path.join(root, name)
			image = adaptive_threshhold_segmentation(img, img_size)
			cv2.imwrite(os.path.join(target, name), image)

def apply_denoise_tv_chambolle(src, image_return_size=224, weight=0.1):
	image = cv2.imread(src)
	image = cv2.resize(image, (image_return_size, image_return_size))
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	image = denoise_tv_chambolle(image, weight, channel_axis=None)*255
	#image = cv2.fromarray(image)
	return image

def apply_denoise_tv_chambolle_directory(src, target, img_size=224, weight=0.1):
	for root, dirs, files in os.walk(src, topdown=False):
		for name in files:
			img = os.path.join(root, name)
			image = apply_denoise_tv_chambolle(img, img_size, weight)
			cv2.imwrite(os.path.join(target, name), image)

# Source: https://learnopencv.com/moving-object-detection-with-opencv/
def remove_background_V2(src, image_return_size=224):
	image = cv2.imread(src)
	image = cv2.resize(image, (image_return_size, image_return_size))
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	coins = skimage.data.coins()
	edges = skimage.feature.canny(coins)
	fill_coins = scipy.ndimage.binary_fill_holes(edges)
	label_objects, nb_labels = scipy.ndimage.label(fill_coins)
	sizes = np.bincount(label_objects.ravel())
	mask_sizes = sizes > 20
	mask_sizes[0] = 0
	coins_cleaned = mask_sizes[label_objects]

	int_arr = np.array(edges, dtype='int')
	int_arr * 255

	return int_arr

def remove_background_V2_directory(src, target, img_size=224):
	for root, dirs, files in os.walk(src, topdown=False):
		for name in files:
			img = os.path.join(root, name)
			image = remove_background_V2(img, img_size)
			cv2.imwrite(os.path.join(target, name), image)

def add_path_to_df(df: pd.DataFrame, path: str, name_column: str = 'Unnamed: 0', set_index: bool = True) -> pd.DataFrame:
	'''This function adds a "path" column which combines the values 
	from the first column with the given path in a new column named 
	"path" '''
	# Combine path with image names and add this data in the path 
	# column in the df
	df['path'] = path
	if(set_index):
		df.rename(columns={name_column: 'name'}, inplace=True)	# Rename Column
		df.set_index('name', inplace=True)
	return df

def get_paths(data_dir):
	result = []
	for root, dirs, files in os.walk(data_dir, topdown=False):
		for name in files:
			img_pth = os.path.join(root, name)
			img_pth = img_pth.replace("/", "\\")
			result.append(img_pth)
	return result

''' Score computation functions ----------------------------------- '''

def compute_assigning_score(
		path_to_solution: str, 
		distance_score_file_name: str, 
		side:bool = True, 
		clusterer: AgglomerativeClustering = AgglomerativeClustering(
			n_clusters=30, linkage='complete', metric='precomputed'), 
		number_of_images: int = -1, 
		distance_function: int = 2) -> tuple[list[int], list[int], int]:
	''' Computes a file that containes the recall, specifity and rand index for the data in "clustered data" in 
	comparison with the data in "path_to_solution".
	Also returns some absolute values connected to this data.
	Side: True = observe, False = reverse '''

	## 1 Clustering
	clustered_df = compute_clustering(
		distance_score_file_name, clusterer=clusterer, 
		number_of_images=number_of_images, 
		distance_function=distance_function)

	## 2 SCORE COMPUTATION

	# Read in solution file 
	temp_solution = pd.read_excel(path_to_solution)

	if side:
		temp_solution = coin_name_format_to_file_name(
			temp_solution, "object_number")
		solution_df = temp_solution[
			["object_number", "final obverse"]].set_index(
				"object_number").rename(columns={
					"final obverse": "final_obverse_GT"})
	else:
		print("Not implemented!")
		return

	clusterings = pd.concat([solution_df, clustered_df], axis=1, join="inner")
	print(clusterings["final_obverse_CL"].nunique())

	# Count number of total pos and total neg and correctly grouped 
	# pairs(true pos and true neg) and amongst them the correctly
	# clustered pairs from our algorithm
	corr_pos = 0
	pos = 0
	corr_neg = 0
	neg = 0
	for index_1 in clusterings.index:
		for index_2 in clusterings.index:
			if (clusterings["final_obverse_GT"][index_1] == 
	   				clusterings["final_obverse_GT"][index_2]):
				if (clusterings["final_obverse_CL"][index_1] == 
						clusterings["final_obverse_CL"][index_2]):
					corr_pos += 1
				pos += 1
			else:
				neg += 1
				if (clusterings["final_obverse_CL"][index_1] != 
						clusterings["final_obverse_CL"][index_2]):
					corr_neg += 1
	
	# Normalize
	diagonal_term = clusterings.shape[0]
	# These are always true, therefore we remove them from the equation
	corr_pos -= diagonal_term
	pos -= diagonal_term

	# Compute results (recall, specifity and rand-index(accuracy))
	if(pos != 0):
		per_pos = corr_pos/pos
	else:
		per_pos = 0
	
	if(neg != 0):
		per_neg = corr_neg/neg
	else:
		per_neg = 0
	rand_index = (corr_neg + corr_pos)/(neg + pos)
	return [per_pos, corr_pos, pos], [per_neg, corr_neg, neg], rand_index

def compute_clustering(
		distance_score_file_name: str, 
		clusterer: AgglomerativeClustering = AgglomerativeClustering(
			n_clusters=30, linkage='complete', metric='precomputed'), 
		number_of_images: int = -1, 
		distance_function: int = 2) -> pd.DataFrame:
	''' Read in the data from "distance_score_file_name", cut off the
	meta-data columns.
	Compute distances.
	Cluster the data according to the distances.
	turn it into a dictionary.'''
	data_file = pd.read_csv(distance_score_file_name)
	
	if(number_of_images > 0):
		distance_score_file = data_file.iloc[
			:number_of_images,:number_of_images+1].set_index("name")
	else:
		distance_score_file = data_file.iloc[:,:-1].set_index("name")
	AC = clusterer

	# Precompute the spearman distances to give to the algorithm
	computed_distances = DistanceFunctionHUB.forward(
		distance_score_file, distance_function)

	clustering = AC.fit(computed_distances) # Clustering

	# Make a dictionary
	clustered_dict = {}
	for a,b in zip(data_file.iloc[:,0], clustering.labels_):
		clustered_dict[a] = b
	clustered_df = pd.DataFrame(clustered_dict.items(), columns=[
		'object_number', 'final_obverse_CL']).set_index("object_number")
	return clustered_df

def coin_name_format_to_file_name(file, column_name):
	'''Makes the filenames from the image names in the .csv files.'''
	for index in file.index:
		file.loc[index, column_name] = \
			file[column_name][index].replace("/", "-") + "A.JPG"
	return file

''' Score computation functions modified to work with the AGLP 
clustering from https://github.com/ClementCornet/Auto-Die-Studies.
Based on "compute_assigning_score" function.'''
def compute_assigning_score_AGLP(
		path_to_solution: str, 
		distance_score_file_name: str, 
		side:bool = True,  
		number_of_images: int = -1, 
		distance_function: int = 2) -> tuple[list[int], list[int], int]:

	## 1 Clustering
	clustered_df = compute_clustering_AGLP(
		distance_score_file_name,
		number_of_images=number_of_images, 
		distance_function=distance_function)
	print(clustered_df)

	## 2 SCORE COMPUTATION

	# Read in solution file 
	temp_solution = pd.read_excel(path_to_solution)

	if side:
		temp_solution = coin_name_format_to_file_name(
			temp_solution, "object_number")
		solution_df = temp_solution[
			["object_number", "final obverse"]].set_index(
				"object_number").rename(columns={
					"final obverse": "final_obverse_GT"})
	else:
		print("Not implemented!")
		return

	clusterings = pd.concat([solution_df, clustered_df], axis=1, join="inner")
	print("Cluster:", clusterings["final_obverse_CL"].nunique())

	# Count number of total pos and total neg and correctly grouped 
	# pairs(true pos and true neg) and amongst them the correctly
	# clustered pairs from our algorithm
	corr_pos = 0
	pos = 0
	corr_neg = 0
	neg = 0
	for index_1 in clusterings.index:
		for index_2 in clusterings.index:
			if (clusterings["final_obverse_GT"][index_1] == 
	   				clusterings["final_obverse_GT"][index_2]):
				if (clusterings["final_obverse_CL"][index_1] == 
						clusterings["final_obverse_CL"][index_2]):
					corr_pos += 1
				pos += 1
			else:
				neg += 1
				if (clusterings["final_obverse_CL"][index_1] != 
						clusterings["final_obverse_CL"][index_2]):
					corr_neg += 1
	
	# Normalize
	diagonal_term = clusterings.shape[0]
	# These are always true, therefore we remove them from the equation
	corr_pos -= diagonal_term
	pos -= diagonal_term

	# Compute results (recall, specifity and rand-index(accuracy))
	if(pos != 0):
		per_pos = corr_pos/pos
	else:
		per_pos = 0
	
	if(neg != 0):
		per_neg = corr_neg/neg
	else:
		per_neg = 0
	rand_index = (corr_neg + corr_pos)/(neg + pos)
	return [per_pos, corr_pos, pos], [per_neg, corr_neg, neg], rand_index

def compute_clustering_AGLP(
		distance_score_file_name: str, 
		number_of_images: int = -1, 
		distance_function: int = 2) -> pd.DataFrame:
	''' Read in the data from "distance_score_file_name", cut off the
	meta-data columns.
	Compute distances.
	Cluster the data according to the distances.
	turn it into a dictionary.'''
	data_file = pd.read_csv(distance_score_file_name)
	
	if(number_of_images > 0):
		distance_score_file = data_file.iloc[
			:number_of_images,:number_of_images+1].set_index("name")
	else:
		distance_score_file = data_file.iloc[:,:-1].set_index("name")
	

	# Precompute the spearman distances to give to the algorithm
	computed_distances = DistanceFunctionHUB.forward(
		distance_score_file, distance_function)

	partition = AGLP_clustering(computed_distances) # Clustering

	# Make a dictionary
	clustered_dict = {}
	for a,b in zip(data_file.iloc[:,0], partition):
		clustered_dict[a] = b
	clustered_df = pd.DataFrame(clustered_dict.items(), columns=[
		'object_number', 'final_obverse_CL']).set_index("object_number")
	return clustered_df

def coin_name_format_to_file_name(file, column_name):
	'''Makes the filenames from the image names in the .csv files.'''
	for index in file.index:
		file.loc[index, column_name] = \
			file[column_name][index].replace("/", "-") + "A.JPG"
	return file

# Copied and modified from https://github.com/ClementCornet/Auto-Die-Studies
def AGLP_clustering(sim):
	"""
	Compute Graph from matches between coins, Label Propagation clustering
	Best graph select upon threshold
	"""
	
	#print(sim.max()) # Should be higher than 1!!!
	if sim.max() <= 1: # Normalize for smaller max values
		dmat = sim.max() - sim
		np.fill_diagonal(dmat, 0)
		np.fill_diagonal(sim, sim.min())
		sim = (exposure.equalize_hist(sim) * 1000.).astype(int)
		cluster_range = sim.max()
		print(sim)

		partitions = [PropagationClustering().fit_predict(scipy.sparse.csr_matrix(sim > th)) for th in tqdm.tqdm(range(int(cluster_range)), desc="Computing partitions for each threshold")]
	else:
		dmat = sim.max() - sim
		np.fill_diagonal(dmat, 0)
		np.fill_diagonal(sim, 0)
		cluster_range = sim.max()
		print(sim)

		partitions = [PropagationClustering().fit_predict(scipy.sparse.csr_matrix(sim > th)) for th in tqdm.tqdm(range(int(cluster_range)), desc="Computing partitions for each threshold")]
	sil = [silhouette_score(dmat, p, metric="precomputed") if (len(set(p)) > 1 and len(set(p))<len(sim)) else 0 for p in tqdm.tqdm(partitions, desc='Computing Silhouettes')]
	print(f'Optimal threshold : {np.argmax(sil)}')
	return partitions[np.argmax(np.array(sil))]


#######################################################################
#
#######################################################################


def create_comparison_file(
		distance_files: list[str], 
		filenames: list[str], 
		start: int, 
		end: int, 
		target_file: str = "overview.csv", 
		number_of_images: int = -1, 
		true_values_file: str = "data_coins/class VI final list.xlsx",
		distance_function: list[int] = [1,1]) -> None:
	'''Creates a file for multiple files to compare in one .csv-file.
	
	The returned dataframe has three columns for positives(Recall), 
	negatives(Specifity) and rand-index.
	Then for each Int from start to end compute their scores and
	puts them into the return-dataframe.
	'''

	# Create the result dataframe
	columns = []
	for i in range(len(distance_files)):
		columns += [filenames[i] + "_Recall", filenames[i] + "_Specifity", 
			  filenames[i] + "_RI"]
	result = pd.DataFrame([], columns=columns)

	for i in range(start, end + 1):
		clusterer = AgglomerativeClustering(
			n_clusters=i, linkage='complete', metric='precomputed')

		# Compute the results for each clustering
		result_i = []
		for j in range(len(distance_files)):
			temp = compute_assigning_score(
				true_values_file, distance_files[j], 
				clusterer=clusterer, number_of_images=number_of_images,
				distance_function=distance_function[j])
			result_i += [temp[0][0], temp[1][0], temp[2]]

		result.loc[i] = result_i

		# Save the file all 10 steps to avoid data loss
		if(i%10 == 9):
			result.to_csv(target_file)
	result.to_csv(target_file)

def analyseClustering(
		matching_file_path: str, 
		solution_file_path: str,
		distance_function: int,
		side:bool = True,
		clusterer: AgglomerativeClustering = AgglomerativeClustering(
			n_clusters=30, linkage='complete', metric='precomputed')
		) -> None:
	
	## Clustering
	clustered_df = compute_clustering(
		matching_file_path, clusterer=clusterer, 
		number_of_images=-1, 
		distance_function=distance_function)


	# Read in solution file 
	temp_solution = pd.read_excel(solution_file_path)

	if side:
		temp_solution = coin_name_format_to_file_name(
			temp_solution, "object_number")
		solution_df = temp_solution[
			["object_number", "final obverse"]].set_index(
				"object_number").rename(columns={
					"final obverse": "final_obverse_GT"})
	else:
		print("Not implemented!")
		return

	clusterings = pd.concat([solution_df, clustered_df], axis=1, join="inner")

	print("Clustering results:")
	print("Number of Clusters(computed):", clusterings["final_obverse_CL"].nunique())
	print("Number of Clusters(ground truth):", clusterings["final_obverse_GT"].nunique())

	perc_clusters = []
	for perc in range(0,11):
		# [Number of clusters achieving at least this amount of equal dies,
		# number of coins in these clusters]
		res = [0, 0]
		for cluster in set(clusterings["final_obverse_CL"].tolist()):
			max_corresp = clusterings.loc[clusterings.final_obverse_CL == cluster, ["final_obverse_GT"]].value_counts().max()
			total = clusterings.loc[clusterings.final_obverse_CL == cluster, "final_obverse_GT"].shape[0]
			# At least perc/10 of the cluster is the same die
			cond = 0
			if(float(max_corresp)/float(total) >= float(perc)/float(10)):cond=1
			res[0] += cond
			res[1] += cond * total
		perc_clusters.append((str(perc*10) + "%", res[0], res[0]/clusterings["final_obverse_CL"].nunique(), res[1], res[1]/clusterings.shape[0]))
	for elem in perc_clusters:
		print(elem[0] + ": " + str(elem[1]), "(" + str(elem[2]) + "), " + str(elem[3]), "(" + str(elem[4]) + ")")
	return None

# True neg wrong atm
def compute_perfect_scores_rand_index(
		path_to_solution: str, filename: str, start: int, end: int) -> None:
	''' Compute the maximum percentages of true positives and negatives
	for each number of clusters. This also maximizes rand index, as 
	true positives and true negatives do not affect each other in this
	scenario: if the number of clusters is bigger or equal than the 
	ground truth, true negatives can always be 100%. If there are less
	or equal the number of clusters, the number of false negatives 
	can always be 100%.'''
	solution_file = pd.read_excel(path_to_solution)

	pos = 0
	neg = 0
	amount = solution_file.shape[0]
	for index_1 in solution_file.index:
		for index_2 in solution_file.index:
			if solution_file["final obverse"][index_1] == \
				solution_file["final obverse"][index_2]:
				pos += 1
			else:
				neg += 1
	occurances = solution_file["final obverse"].\
		value_counts(ascending=True)

	pos -= amount

	print("Pos & Neg: ", pos, neg)
	#print(occurances)

	result = pd.DataFrame([], columns=[
		"ri_perfect_pos", "ri_perfect_neg", "ri_perfect_rand_index"])
	for num_clusters in range(start, end + 1):
		
		removable_occurances = occurances.copy()
		max_true_pos = computeTruePositives(
			pos, removable_occurances=removable_occurances, 
			number_of_clusters=num_clusters)

		removable_occurances = occurances.copy()
		max_true_neg = computeTrueNegatives(
			neg, removable_occurances=removable_occurances, 
			number_of_clusters=num_clusters, 
			solution=solution_file["final obverse"])

		result.loc[num_clusters] = [
			max_true_pos/pos, max_true_neg/neg,
			(max_true_pos + max_true_neg)/(pos + neg)]
		print(num_clusters, "/", end)
		if(num_clusters%10 == 0):
			result.to_csv(filename)
	result.to_csv(filename)

def computeTruePositives(
		pos_total: int, removable_occurances: int, 
		number_of_clusters: int) -> int:
	'''For each cluster more than the amount of ground-truth clusters,
	take an element from the smallest cluster (with at least 2 
	elements) and put it in it's own cluster'''
	# The number of true positives stays max, until we use more 
	# clusteres than actually exist.
	range_ind = max(number_of_clusters - len(removable_occurances), 0)

	for i in range(range_ind):
			if len(removable_occurances) < 1: # Failsafe
				print("compute_perfect_scores_rand_index: \
		  			To early finished.")
				print(removable_occurances)
				break

			# Remove sets with less then 2 items, because you cannot 
			# make 2 sets from them
			if(removable_occurances.iloc[0] < 2): 
				removable_occurances = removable_occurances.iloc[1:]
			
			# If an item gets removed from a set, all remaining items 
			# lose 1 true positive and the removed item loses a for all
			# remaining items. Hence 2 * (len-1)
			removable_occurances.iloc[0] -= 1
			pos_total -= 2 * removable_occurances.iloc[0]
			
			### print(removable_occurances.iloc[0])
	max_true_pos = pos_total
	return max_true_pos

# Brute Force!
def computeTrueNegatives(max_true_neg: int, removable_occurances: int, 
						 number_of_clusters: int, solution):
	if number_of_clusters > len(removable_occurances):
		return max_true_neg
	else:
		max_solution = 0
		prop_solution = pd.DataFrame(1, index=np.arange(
			removable_occurances.sum()), columns=["final obverse"])
		while(prop_solution.sum()["final obverse"] < 
				number_of_clusters * len(prop_solution)):
			i = 0
			prop_solution.loc[i, "final obverse"] += 1
			while(prop_solution.loc[i, "final obverse"] > \
		 			number_of_clusters):
				prop_solution.loc[i, "final obverse"] = 0
				prop_solution.loc[i+1, "final obverse"] += 1
				i += 1
			number_true_negatives = count_true_negatives(solution, 
												prop_solution)
			if(number_true_negatives > max_solution):
				max_solution = number_true_negatives
	return max_solution

def count_true_negatives(clustering_1, clustering_2):
	''' Make matrices of pairs having different clusters in each 
	clustering get those, which are different in both clusterings
	(&). Than sum the occurances up to get the amount'''
	amount = (clustering_1.ne(clustering_1.T)&
		   clustering_2.ne(clustering_2.T)).sum().iloc[:-1].sum()
	return amount

def clustering_differences(clustering_1, clustering_2):
	''' Looks at two clusterings and returns statistics about the
	differences and similarities: - Matrix, which images are clustered
	in the same way (in both clusterings ind the same or in both in 
	different clusters) or in different ways.'''
	same_clustering_matrix = pd.DataFrame(
		0, index=range(clustering_1.size), 
		columns=range(clustering_1.size))
	
	clustering_1_same_clustering_matrix = pd.DataFrame(
		0, index=range(clustering_1.size), 
		columns=range(clustering_1.size))
	
	clustering_2_same_clustering_matrix = pd.DataFrame(
		0, index=range(clustering_1.size), 
		columns=range(clustering_1.size))

	# Go through all images
	for ind1 in range(0, clustering_1.size):
		for ind2 in range(0, clustering_1.size):
			if (clustering_1.iloc[ind1]["final_obverse_CL"] == \
	   				clustering_1.iloc[ind2]["final_obverse_CL"]) and \
					(clustering_2.iloc[ind1]["final_obverse_CL"] == \
	  				clustering_2.iloc[ind2]["final_obverse_CL"]):
				same_clustering_matrix.iloc[ind1, ind2] = 1
				clustering_1_same_clustering_matrix.iloc[ind1, ind2] = \
					clustering_2_same_clustering_matrix.iloc[ind1, ind2] = 1
			elif (clustering_1.iloc[ind1]["final_obverse_CL"] != \
		 			clustering_1.iloc[ind2]["final_obverse_CL"]) and \
					(clustering_2.iloc[ind1]["final_obverse_CL"] != \
	  				clustering_2.iloc[ind2]["final_obverse_CL"]):
				same_clustering_matrix.iloc[ind1, ind2] = 1
				clustering_1_same_clustering_matrix.iloc[ind1, ind2] = \
					clustering_2_same_clustering_matrix.iloc[ind1, ind2] = 0
			else:
				if clustering_1.iloc[ind1]["final_obverse_CL"] == \
					clustering_1.iloc[ind2]["final_obverse_CL"]:
					clustering_1_same_clustering_matrix.iloc[ind1, ind2] = 1
					clustering_2_same_clustering_matrix.iloc[ind1, ind2] = 0
				else:
					clustering_1_same_clustering_matrix.iloc[ind1, ind2] = 0
					clustering_2_same_clustering_matrix.iloc[ind1, ind2] = 1
				same_clustering_matrix.iloc[ind1, ind2] = 0

	return same_clustering_matrix, [
		clustering_1_same_clustering_matrix, 
		clustering_2_same_clustering_matrix]

#######################################################################
#
# Distance score computations
#
#######################################################################

class DistanceFunctionHUB():
	methods = ["No_Distance_Function", "Spearman", "Pearson", "Cosine", 
			"scipy_stats_linregress", "scipy_stats_pointbiserialr",
			"scipy_stats_kendalltau", "scipy_stats_somersd",
			"scipy_spatial_braycurtis", "scipy_spatial_canberra",
			"scipy_spatial_chebyshev", "scipy_spatial_cityblock", 
			"scipy_spatial_correlation",
			"scipy_spatial_euclidean", "scipy_spatial_jensenshannon",
			"scipy_spatial_minkowski", "scipy_spatial_seuclidean", 
			"scipy_spatial_sqeuclidean", "scipy_spatial_yule"]

	@staticmethod
	def forward(input_matrix: pd.DataFrame, method: int = 0) -> np.ndarray:
		return DistanceFunctionHUB.method_functions[method](input_matrix)

	@staticmethod
	def no_distance(input_matrix: pd.DataFrame) -> np.ndarray:
		return input_matrix.to_numpy()

	@staticmethod
	def spearman_distance_score(input_matrix: pd.DataFrame) -> np.ndarray:
		'''Computes normalized spearman distances.
		Method 0 in this class.'''
		distance_score_file = pd.DataFrame(normalize(input_matrix))
		#print(distance_score_file.shape)

		spearman_distances = Orange.distance.SpearmanR(
			distance_score_file, normalize=True)

		# The same implementation with scipy
		# distance_score_file = np.array(distance_score_file)
		# spearman_distances = np.zeros_like(distance_score_file)
		# for i in range(np.shape(spearman_distances)[0]):
		# 	for j in range(np.shape(spearman_distances)[1]):
		# 		spearman_distances[i,j] = scipy.stats.spearmanr(
		# 			np.array(distance_score_file[i]), 
		# 			np.array(distance_score_file[j])).statistic
		# spearman_distances = (1 - spearman_distances) / 2

		spearman_distances = np.nan_to_num(spearman_distances)

		return spearman_distances

	@staticmethod
	def pearson_distance_score(input_matrix: pd.DataFrame) -> np.ndarray:
		'''Computes normalized pearson distances.
		Method 1 in this class.'''
		distance_score_file = pd.DataFrame(normalize(input_matrix))
		#print(distance_score_file.shape)

		# Compute pearson distances (correlation/standarddeviations)
		# and then move the scope to the interval [0,1]
		#pearson_distances = np.corrcoef(distance_score_file)
		#pearson_distances = np.nan_to_num(pearson_distances)
		#standard_dev = np.array(np.std(distance_score_file, axis=0)).\
		#	reshape((distance_score_file.shape[0]))
		#standard_dev_T = standard_dev[np.newaxis].T
		#pearson_distances = np.array(pearson_distances)
		#pearson_distances = np.multiply(
		#	np.multiply(pearson_distances, standard_dev), standard_dev_T)
		
		# Scipy implementation, which is much slower and not multiplied by
		# the std, therefore worse.
		# distance_score_file = np.array(distance_score_file)
		# pearson_distances = np.zeros_like(distance_score_file)
		# for i in range(np.shape(pearson_distances)[0]):
		# 	for j in range(np.shape(pearson_distances)[1]):
		# 		pearson_distances[i,j] = scipy.stats.pearsonr(
		# 			np.array(distance_score_file[i]), 
		# 			np.array(distance_score_file[j])).statistic

		# Interval correction to [0,1]
		#pearson_distances = (1 - pearson_distances) / 2
		#pearson_distances = np.nan_to_num(pearson_distances)

		pearson_distances = Orange.distance.PearsonR(
			distance_score_file, normalize=True)
		
		return pearson_distances

	@staticmethod
	def cosine_distance_score(input_matrix: pd.DataFrame) -> np.ndarray:
		'''Computes normalized cosine distances.
		Method 2 in this class.'''
		distance_score_file = pd.DataFrame(normalize(input_matrix))

		cosin_distances = np.zeros_like(distance_score_file)
		for i in range(np.shape(cosin_distances)[0]):
			for j in range(np.shape(cosin_distances)[1]):
				# Normalization via squaring of the value.
				cosin_distances[i,j] = scipy.spatial.distance.cosine(
					distance_score_file[i], distance_score_file[j])**2
				
				# Own implementation that achieves the same results
				#a = np.array(distance_score_file[i])
				#b = np.array(distance_score_file[j])
				#cosin_distances[i, j] = (1 - (np.dot(a, b)/
				#					(np.linalg.norm(a)*
				#					np.linalg.norm(b))))**2
		
		cosin_distances = np.nan_to_num(cosin_distances)

		return cosin_distances

	# Scipy.stats functions
	@staticmethod
	def scipy_stats_linregress(input_matrix: pd.DataFrame) -> np.ndarray:
		'''Computes the linregress distances.'''
		distance_score_file = pd.DataFrame(normalize(input_matrix))

		testing_distances = np.zeros_like(distance_score_file)
		for i in range(np.shape(testing_distances)[0]):
			for j in range(np.shape(testing_distances)[1]):
				# The normalization may not be necessairy, but it doesn't
				# change the results either in the tests.
				testing_distances[i,j] = scipy.stats.linregress(np.array(distance_score_file[i]), np.array(distance_score_file[j])).slope
		testing_distances = (1 - testing_distances)/2
		
		testing_distances = np.nan_to_num(testing_distances)

		return testing_distances 

	@staticmethod
	def scipy_stats_pointbiserialr(input_matrix: pd.DataFrame) -> np.ndarray:
		'''Computes the linregress distances.'''
		distance_score_file = pd.DataFrame(normalize(input_matrix))

		testing_distances = np.zeros_like(distance_score_file)
		for i in range(np.shape(testing_distances)[0]):
			for j in range(np.shape(testing_distances)[1]):
				# The normalization may not be necessairy, but it doesn't
				# change the results either in the tests.
				testing_distances[i,j] = scipy.stats.pointbiserialr(np.array(distance_score_file[i]), np.array(distance_score_file[j])).statistic
		testing_distances = (1 - testing_distances)/2
		
		testing_distances = np.nan_to_num(testing_distances)

		return testing_distances
	
	@staticmethod
	def scipy_stats_kendalltau(input_matrix: pd.DataFrame) -> np.ndarray:
		'''Used for testing functions drom scipy.stats.'''
		distance_score_file = pd.DataFrame(normalize(input_matrix))

		testing_distances = np.zeros_like(distance_score_file)
		for i in range(np.shape(testing_distances)[0]):
			for j in range(np.shape(testing_distances)[1]):
				# The normalization may not be necessairy, but it doesn't
				# change the results either in the tests.
				testing_distances[i,j] = scipy.stats.kendalltau(np.array(distance_score_file[i]), np.array(distance_score_file[j]), variant='b').statistic
		testing_distances = (1 - testing_distances)/2
		
		testing_distances = np.nan_to_num(testing_distances)

		return testing_distances
	
	@staticmethod
	def scipy_stats_somersd(input_matrix: pd.DataFrame) -> np.ndarray:
		'''Used for testing functions drom scipy.stats.'''
		distance_score_file = pd.DataFrame(normalize(input_matrix))

		testing_distances = np.zeros_like(distance_score_file)
		for i in range(np.shape(testing_distances)[0]):
			for j in range(np.shape(testing_distances)[1]):
				# The normalization may not be necessairy, but it doesn't
				# change the results either in the tests.
				testing_distances[i,j] = scipy.stats.somersd(np.array(distance_score_file[i]), np.array(distance_score_file[j])).statistic
		testing_distances = (1 - testing_distances)/2
		
		testing_distances = np.nan_to_num(testing_distances)

		return testing_distances 

	@staticmethod
	def scipy_stats_multiscale_graphcorr(input_matrix: pd.DataFrame) -> np.ndarray:
		'''Takes too long to compute and is therefore not used.'''
		distance_score_file = pd.DataFrame(normalize(input_matrix))

		testing_distances = np.zeros_like(distance_score_file)
		for i in range(np.shape(testing_distances)[0]):
			for j in range(np.shape(testing_distances)[1]):
				# The normalization may not be necessairy, but it doesn't
				# change the results either in the tests.
				testing_distances[i,j] = scipy.stats.multiscale_graphcorr(np.array(distance_score_file[i]), np.array(distance_score_file[j])).statistic
		testing_distances = (1 - testing_distances)/2
		
		testing_distances = np.nan_to_num(testing_distances)

		return testing_distances
	
	# Scipy.spacial functions
	@staticmethod
	def scipy_spatial_braycurtis(input_matrix: pd.DataFrame) -> np.ndarray:
		'''Used for testing functions drom scipy.spatial.distance.'''
		distance_score_file = pd.DataFrame(normalize(input_matrix))

		distances = scipy.spatial.distance.squareform(
			scipy.spatial.distance.pdist(distance_score_file, 'braycurtis'))
		
		distances = np.nan_to_num(distances)

		return distances

	@staticmethod
	def scipy_spatial_canberra(input_matrix: pd.DataFrame) -> np.ndarray:
		'''Used for testing functions drom scipy.spatial.distance.'''
		distance_score_file = pd.DataFrame(normalize(input_matrix))

		distances = scipy.spatial.distance.squareform(
			scipy.spatial.distance.pdist(distance_score_file, 'canberra'))
		
		distances = np.nan_to_num(distances)

		return distances
	
	@staticmethod
	def scipy_spatial_chebyshev(input_matrix: pd.DataFrame) -> np.ndarray:
		'''Used for testing functions drom scipy.spatial.distance.'''
		distance_score_file = pd.DataFrame(normalize(input_matrix))

		distances = scipy.spatial.distance.squareform(
			scipy.spatial.distance.pdist(distance_score_file, 'chebyshev'))
		
		distances = np.nan_to_num(distances)

		return distances 

	@staticmethod
	def scipy_spatial_cityblock(input_matrix: pd.DataFrame) -> np.ndarray:
		'''Used for testing functions drom scipy.spatial.distance.'''
		distance_score_file = pd.DataFrame(normalize(input_matrix))

		distances = scipy.spatial.distance.squareform(
			scipy.spatial.distance.pdist(distance_score_file, 'cityblock'))
		
		distances = np.nan_to_num(distances)

		return distances
	
	@staticmethod
	def scipy_spatial_correlation(input_matrix: pd.DataFrame) -> np.ndarray:
		'''Used for testing functions drom scipy.spatial.distance.'''
		distance_score_file = pd.DataFrame(normalize(input_matrix))

		distances = scipy.spatial.distance.squareform(
			scipy.spatial.distance.pdist(distance_score_file, 'correlation'))
		
		distances = np.nan_to_num(distances)

		return distances 
	
	@staticmethod
	def scipy_spatial_euclidean(input_matrix: pd.DataFrame) -> np.ndarray:
		'''Used for testing functions drom scipy.spatial.distance.'''
		distance_score_file = pd.DataFrame(normalize(input_matrix))

		distances = scipy.spatial.distance.squareform(
			scipy.spatial.distance.pdist(distance_score_file, 'euclidean'))
		
		distances = np.nan_to_num(distances)

		return distances

	@staticmethod
	def scipy_spatial_jensenshannon(input_matrix: pd.DataFrame) -> np.ndarray:
		'''Used for testing functions drom scipy.spatial.distance.'''
		distance_score_file = pd.DataFrame(normalize(input_matrix))

		distances = scipy.spatial.distance.squareform(
			scipy.spatial.distance.pdist(distance_score_file, 'jensenshannon'))
		
		distances = np.nan_to_num(distances)

		return distances
	
	@staticmethod
	def scipy_spatial_mahalanobis(input_matrix: pd.DataFrame) -> np.ndarray:
		'''"The number of observations (100) is too small;"
		Therefore doesn't work in our context for quadratic matrices.'''
		distance_score_file = pd.DataFrame(normalize(input_matrix))

		distances = scipy.spatial.distance.squareform(
			scipy.spatial.distance.pdist(distance_score_file, 'mahalanobis'))
		
		distances = np.nan_to_num(distances)

		return distances

	@staticmethod
	def scipy_spatial_minkowski(input_matrix: pd.DataFrame) -> np.ndarray:
		'''Used for testing functions drom scipy.spatial.distance.'''
		distance_score_file = pd.DataFrame(normalize(input_matrix))

		distances = scipy.spatial.distance.squareform(
			scipy.spatial.distance.pdist(distance_score_file, 'minkowski'))
		
		distances = np.nan_to_num(distances)

		return distances
	
	@staticmethod
	def scipy_spatial_seuclidean(input_matrix: pd.DataFrame) -> np.ndarray:
		'''Used for testing functions drom scipy.spatial.distance.'''
		distance_score_file = pd.DataFrame(normalize(input_matrix))

		distances = scipy.spatial.distance.squareform(
			scipy.spatial.distance.pdist(distance_score_file, 'seuclidean'))
		
		distances = np.nan_to_num(distances)

		return distances
	
	@staticmethod
	def scipy_spatial_sqeuclidean(input_matrix: pd.DataFrame) -> np.ndarray:
		'''Used for testing functions drom scipy.spatial.distance.'''
		distance_score_file = pd.DataFrame(normalize(input_matrix))

		distances = scipy.spatial.distance.squareform(
			scipy.spatial.distance.pdist(distance_score_file, 'sqeuclidean'))
		
		distances = np.nan_to_num(distances)

		return distances
	
	@staticmethod
	def scipy_spatial_yule(input_matrix: pd.DataFrame) -> np.ndarray:
		'''Used for testing functions drom scipy.spatial.distance.'''
		distance_score_file = pd.DataFrame(normalize(input_matrix))

		distances = np.zeros_like(distance_score_file)
		for i in range(np.shape(distances)[0]):
			for j in range(np.shape(distances)[1]):
				# The normalization may not be necessairy, but it doesn't
				# change the results either in the tests.
				distances[i,j] = scipy.spatial.distance.yule(
					np.array(distance_score_file[i]), 
					np.array(distance_score_file[j]))
		
		distances = np.nan_to_num(distances)

		return distances

	method_functions = [no_distance, spearman_distance_score, 
					 pearson_distance_score, cosine_distance_score, 
					 scipy_stats_linregress, scipy_stats_pointbiserialr,
					 scipy_stats_kendalltau, scipy_stats_somersd,
					 scipy_spatial_braycurtis, scipy_spatial_canberra,
					 scipy_spatial_chebyshev, scipy_spatial_cityblock, 
					 scipy_spatial_correlation, scipy_spatial_euclidean, 
					 scipy_spatial_jensenshannon, scipy_spatial_minkowski, 
					 scipy_spatial_seuclidean, 
					 scipy_spatial_sqeuclidean, scipy_spatial_yule]
# Path to the folder, where the images of the coins are. Use '\\' for 
# Windows-paths and '/' for Linux-paths.
# Example for Windows: "C:\\Folder\\...\\images\\"
images_directory: str = "/home/student_01/Projects/Datasets/dataset_coins_GL_reverse/"

# Path were the preprocessed images and the results will be stored.
# Same rules for the path as for "images_path"
results_directory: str = "/home/student_01/Projects/Datasets/coins_GL_reverse_results/"
# Methods to use for matching and distance comnputation. The different
# functions corresponding to the numbers are shown at the end of this file.
matching_computation_method: int = 4
distance_computation_method: int = 2

# Names of the resulting files.
# The file 'matching_file_name' contains the results of the matching.
# The file 'clustering_file_name' contains the clustering, which can
# be viewed in Orange.
matching_file_name: str = "matching.csv"
clustering_file_name: str = "clustering_reverse_4_2_50.csv"

# The number of clusters, that the coins should be distributed to.
number_of_clusters: int = 50

####################################################################################
# Overviews

# Matching computation methods
# 0 "ORB",
# 1 "matcher nn",
# 2 "matcher mnn",
# 3 "matcher snn",
# 4 "matcher smnn",
# 5 "matcher fginn",
# 6 "matcher AdaLAM",
# 7 "matcher lightglue",
# 8 "matcher LoFTR",
# 9 "detector gtff_response",
#10 "detector dog_response_single",
#11 "kornia_descriptor_Dense_Sift_descriptor",
#12 "kornia_descriptor_SIFT_descriptor",
#13 "kornia_descriptor_MKDDescriptor",
#14 "kornia_descriptor_HardNet_descriptor",
#15 "kornia_descriptor_Hardnet8_descriptor",
#16 "kornia_descriptor_HyNet_descriptor",
#17 "kornia_descriptor_TFeat_descriptor",
#18 "kornia_descriptor_SOSNet_descriptor",
#19 "dete_and_dest SOLD2_detector",
#20 "dete_and_dest DeDoDe",
#21 "dete_and_dest DISK",
#22 "dete_and_dest SIFTFeature",
#23 "dete_and_dest GFTTAffNetHardNet",
#24 "dete_and_dest KeyNetAffNetHardNet",
#25 "OpenCV 2nn",
#26 "Testing",
#27 "smnn_abs_count",
#28 "kornia nn ORB",
#29 "OpenCV smnn",
#30 "smnn DISK with estimator USAC_ACCURATE"

# Distance compuation functions
# 0 "No_Distance_Function", 
# 1 "Spearman", 
# 2 "Pearson", 
# 3 "Cosine", 
# 4 "scipy_stats_linregress", 
# 5 "scipy_stats_pointbiserialr",
# 6 "scipy_stats_kendalltau", 
# 7 "scipy_stats_somersd",
# 8 "scipy_spatial_braycurtis", 
# 9 "scipy_spatial_canberra",
#10 "scipy_spatial_chebyshev", 
#11 "scipy_spatial_cityblock", 
#12 "scipy_spatial_correlation",
#13 "scipy_spatial_euclidean", 
#14 "scipy_spatial_jensenshannon",
#15 "scipy_spatial_minkowski", 
#16 "scipy_spatial_seuclidean", 
#17 "scipy_spatial_sqeuclidean", 
#18 "scipy_spatial_yule"
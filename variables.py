# Path to the folder, where the images of the coins are. Use '\\' for 
# Windows-paths and '/' for Linux-paths.
# Example for Windows: "C:\\Folder\\...\\images\\"
images_directory: str = "D:\\Studium\\Informatik\\9. Semester (Master)\\Masterarbeit\\Weitere Muenzsets\\GL_Bilder\\reverse\\"  # "D:\\Studium\\Informatik\\9. Semester (Master)\\Masterarbeit\\Weitere Muenzsets\\GL_Bilder\\reverse\\"

# Path were the preprocessed images and the results will be stored.
# Same rules for the path as for "images_path"
results_directory: str = "D:\\Studium\\Informatik\\9. Semester (Master)\\Masterarbeit\\Weitere Muenzsets\\GL_Bilder\\reverse_results\\" # "D:\\Studium\\Informatik\\9. Semester (Master)\\Masterarbeit\\Weitere Muenzsets\\GL_Bilder\\reverse_results\\"

# Methods to use for matching and distance comnputation. The different
# functions corresponding to the numbers are at the end of the
# "kornia_matcher.py" file under 'class MatchingHandler'
matching_computation_method: int = 4
distance_computation_method: int = 2

# Names of the resulting files.
# The file 'matching_file_name' contains the results of the matching.
# The file 'clustering_file_name' contains the clustering, which can
# be viewed in Orange.
matching_file_name: str = "matching.csv"
clustering_file_name: str = "clustering_obverse_4_2_50.csv"

# The number of clusters, that the coins should be distributed to.
number_of_clusters: int = 50
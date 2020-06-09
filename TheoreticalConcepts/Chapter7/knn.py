#import the necessary packages

# sklearn library
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Helper Codes
from scripts.preprocessing import SimplePreprocessor
from scripts.datasets import SimpleDatasetLoader

# Other imports
from imutils import paths	# Image Path Functionality
import argparse		# Command line arguments


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help = "path to input dataset")
ap.add_argument("-k", "--neighbours", type=int, default=1, help="# of nearest neighbors for classification")
ap.add_argument("-j", "--jobs", type=int, default=-1, help="# of jobs for k-NN distance(-1 uses all available cores)")
args = vars(ap.parse_args())

# Step 1: Gather our Dataset

# grab the list of images that we'll be describing
print("[INFO] loading images...")
# print(args["dataset"])
imagePaths = list(paths.list_images(args["dataset"]))
# print(imagePaths)

# initialize the image preprocessor, load the dataset from disk,
# and reshape the data matrix
sp = SimplePreprocessor(32,32) # Resize to 32x32
sdl = SimpleDatasetLoader(preprocessors=[sp]) # Initialize Data Loader Class with sp
(data, labels) = sdl.load(imagePaths, verbose=500)
data = data.reshape((data.shape[0], 3072))

# show some information on memory consumption of the images
print("[INFO] features matrix: {:.1f}MB".format(data.nbytes/(1024*1024.0)))


# Step 2: Split the Dataset

# encode the labels as integers
le = LabelEncoder()
labels = le.fit_transform(labels)

# partition the dat into training and testing splits using 75% of
# the data for training and the remaining 25% for testing.
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)


# Step 3: Train the Classifier

# train and evaluate a k-NN classfier on the raw pixel intensities
print("[INFO] evaluating a k-NN classifier...")
model = KNeighborsClassifier(n_neighbors=args["neighbours"],n_jobs=args["jobs"])
model.fit(trainX, trainY)


# Step 4: Evaluate
print(classification_report(testY, model.predict(testX), target_names=le.classes_))
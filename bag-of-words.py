import nltk
import pandas as pd
from nltk.corpus import stopwords # Import the stop word list
import re
import string
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
printable = set(string.printable)

# Download text data sets, including stop words
# you can give any csv which has blog and category column
train = pd.read_csv("blogs.csv", header=0, \
                    delimiter=",", quoting=2)
nltk.download()

stopwords.words("english")
print "total [row,column] in csv =>",train.shape
print "list of category =>",train.columns.values
# Remove stop words from "words"


def blog_to_words( blog ):
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw movie review), and
    # the output is a single string (a preprocessed movie review)
    #
    # 1. Remove HTML
    # review_text = BeautifulSoup(raw_review).get_text()
    review_text = str(blog)
    # 2. Remove non-letters
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)
    #
    # 3. Convert to lower case, split into individual words
    lower_case = letters_only.lower()  # Convert to lower case
    words = lower_case.lower().split()
    #
    # 4. In Python, searching a set is much faster than searching
    #    a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))
    #
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]
    #
    # 6. Join the words back into one string separated by space,
    # and return the result.
    return ( " ".join( meaningful_words ))


# clean_blog = blog_to_words(train["blog"][0] )
# print clean_blog

# Get the number of reviews based on the dataframe column size
num_blogs = train["blog"].size

# Loop over each review; create an index i that goes from 0 to the length
# of the movie review list
print "Cleaning and parsing the training set blogs..."
clean_train_blogs = []
for i in xrange( 0, num_blogs ):
    # If the index is evenly divisible by 1000, print a message
    if (i+1) %50 == 0:
        print "blog %d of %d" % ( i+1, num_blogs )
    clean_train_blogs.append(blog_to_words(train["blog"][i]))


print "Creating the bag of words..."

# Initialize the "CountVectorizer" object, which is scikit-learn's
# bag of words tool.
vectorizer = CountVectorizer(analyzer = "word",\
                             tokenizer = None,\
                             preprocessor = None,\
                             stop_words = None,\
                             max_features = 5000)

# fit_transform() does two functions: First, it fits the model
# and learns the vocabulary; second, it transforms our training data
# into feature vectors. The input to fit_transform should be a list of
# strings.
train_data_features = vectorizer.fit_transform(clean_train_blogs)

# Numpy arrays are easy to work with, so convert the result to an
# array
train_data_features = train_data_features.toarray()

print "training data features shape",train_data_features.shape

# Take a look at the words in the vocabulary
vocab = vectorizer.get_feature_names()
# print vocab

print "Training the random forest..."

# Initialize a Random Forest classifier with 100 trees
forest = RandomForestClassifier(n_estimators = 100)

# Fit the forest to the training set, using the bag of words as
# features and the sentiment labels as the response variable
#
# This may take a few minutes to run
forest = forest.fit( train_data_features, train["category"])

print("saving forest onject ... in forest.pickle")
with open('forest.pickle', 'wb') as outfile:
    pickle.dump(forest, outfile)
    outfile.close()

print("saving vocab object ... in vocab.pickle")
with open('vocab.pickle','wb') as outfile:
    pickle.dump(vocab,outfile)
    outfile.close()


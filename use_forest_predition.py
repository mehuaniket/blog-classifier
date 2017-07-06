# coding=utf-8
from nltk.corpus import stopwords # Import the stop word list
import re
import string
import pickle
from sklearn.feature_extraction.text import CountVectorizer

printable = set(string.printable)

def blog_to_words( blog ):
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw movie review), and
    # the output is a single string (a preprocessed movie review)
    #
    # 1. Remove HTML
    #
    review_text = str(blog)
    # 2. Remove non-letters
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)
    #
    # 3. Convert to lower case, split into individual words
    lower_case = letters_only.lower()  # Convert to lower case
    words = lower_case.lower().split()
    #
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))
    #
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]
    #
    # 6. Join the words back into one string separated by space,
    # and return the result.
    return " ".join(meaningful_words)

vocabulary_to_load = pickle.load(open("vocab.pickle", 'r'))
vectorizer = CountVectorizer(ngram_range=(5000,
                                        5000), min_df=1, vocabulary=vocabulary_to_load)
vectorizer._validate_vocabulary()

try:
    f = open('forest.pickle', 'rb')
    forest = pickle.load(f)
    f.close()

except:
    print "Prepare the classifier using - train_classifier.py and then try to use."

clean_test_blog = u"""Many government agencies have media accounts that they used for community outreach. TSA has done a great job trying to engage the public. So much so that Rolling Stone has ranked the Transportation Security Administration (TSA) Instagram N0. 4. It is “sandwiched between badgalriri (Rihanna) and Beyoncé on RollingStone.com’s 100 best Instagram accounts”. The TSA postings were mostly devoted to photos of items that they have confiscated from passengers’ luggage which Rolling Stone found fascinating, entertaining and terrifying.
TSA Instagram has more than half a million followers with over 150,000 comments to its posts and more than 2 million likes for its images. It’s most popular Instagram image is a life-size prop dummy from the “Texas Chainsaw Massacre” movie. The image received more than 10,000 likes.
Many government agencies have media accounts that they used for community outreach. TSA has done a great job trying to engage the public. So much so that Rolling Stone has ranked the Transportation Security Administration (TSA) Instagram N0. 4. It is “sandwiched between badgalriri (Rihanna) and Beyoncé on RollingStone.com’s 100 best Instagram accounts”. The TSA postings were mostly devoted to photos of items that they have confiscated from passengers’ luggage which Rolling Stone found fascinating, entertaining and terrifying. TSA Instagram has more than half a million followers with over 150,000 comments to its posts and more than 2 million likes for its images. It’s most popular Instagram image is a life-size prop dummy from the “Texas Chainsaw Massacre” movie. The image received more than 10,000 likes."""

clean_test_filter_blog = filter(lambda x: x in printable, clean_test_blog)
clean_test_blog = blog_to_words(clean_test_filter_blog)

# Get a bag of words for the test set, and convert to a numpy array
test_data_features = vectorizer.transform([clean_test_blog])
test_data_features = test_data_features.toarray()
print test_data_features
# Use the random forest to make sentiment label predictions
result = forest.predict(test_data_features)
print result

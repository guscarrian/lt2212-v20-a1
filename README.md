# LT2212 V20 Assignment 1


**PART 1: Convert the data into a DataFrame**

In this fist part, two folders and and integer are taken as input and we get a DataFrame where NaNs are replaced with zeros. Two helper functions are also implemented to run part1_load:

- dict_lists: it creates a dictionary, where the keys are classnames/filenames and the values are lists of tokenized words from each file found whithin the directory. For tokenizing, the text is split by spaces (.split(' ')), integers and punctuation are filtered out (.isalpha()) and all words are in lowercase (.lower()).

- split_and_counter: it creates a list of dictionaries. Each key of every dictionary within the list is a string which represents the classname, filename and words found in the file. Regarding values, we find the actual directory name (crude or grain), filenames (articlexxx.txt) and occurrences that each word appears in the file.



**PART 4: tf-idf & visualize**

While the top terms in the bar chart from part 2 mainly correspond to function words like prepositions ('to', 'of', 'in', 'for') or articles ('the', 'a'), we see an increase of content words in the bar chart for part 4 ('tonnes', 'wheat', 'oil'). The reason for this is that, after applying tf-idf, the number of function words is filtered out, which translates into a higher amount of content words whithin the top m term frequencies. In addition, it is interesting to note that whilst the top terms before tf-idf was performed were occuring in both 'crude' and 'grain' with more or less the same frequency (except for 'oil'), there are some specific content words for 'crude' and 'grain' respectively. That is, 'tonnes' and 'wheat' are much more frequent in the 'grain' class, however, 'oil' is most frequent in 'crude'.


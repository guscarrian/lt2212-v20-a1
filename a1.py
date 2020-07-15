import os
import sys
import pandas as pd
import numpy as np
import numpy.random as npr


# MY IMPORTS
import glob
import matplotlib.pyplot as plt
import math
from collections import Counter



# HELPER FUNCTIONS

def dict_lists(directory):
    '''                    
    The function creates a dictionary, where the keys are classnames/filenames and the values are lists
    of tokenized words from each file found whithin the directory. Integers and punctuation are filtered out and
    all words are in lowercase.
    
    Arg:
        directory - Directory with the files we want to work with.
    Returns:
        Dictionary of lists.
        Ex: {'crude/article018.txt': ['triton', 'lt', 'says', 'paris' ...]}
        
    '''
    
    dir_file = glob.glob('{}/*.txt'.format(directory)) #glob.glob returns the list of files with their full path
    my_dict = {}
    for file in dir_file:
        words = [] 
        with open(file, "r") as f:
            for line in f.readlines():
                for word in line.split(' '):
                    if word.isalpha():
                        words.append(word.lower())
                        #print(words)
                        my_dict[file] = words    
    
    return my_dict
    #print(my_dict)
    
dict_lists('crude')



def split_and_counter(directory):

    '''
    The function creates a list of dictionaries. Each key of every dictionary
    within the list is a string which represents the classname, filename and words 
    found in the file. Regarding values, we find the actual directory name
    (crude or grain), filenames (articlexxx.txt) and occurrences that each word appears
    in the file.
    
    Arg:
        directory - Directory with the files we want to work with.
    Returns:
        List of dictionaries.
        Ex: [{'classname': 'crude', 'filename': 'article018.txt', 'triton': 4, 'lt': 3,},
             {'classname': 'crude', 'filename': 'article021.txt', 'kuwaiti': 1, 'daily': 3}]
    
    '''

    dir = dict_lists(directory)
    final_list = []

    #getting the path (classname/filename) and splitting it (classname - filename)
    for i in dir.items():
        path = i[0] #classname/filename.txt
        class_file = path.split('/') #splitting path
        
        #adding classname and filenames to the new dict
        word_counter_dict = {}
        word_counter_dict['classname'] = class_file[0]
        word_counter_dict['filename'] = class_file[1]
        words = i[1]

        #word counter + append to list
        counting_words = Counter(words)
        word_counter_dict.update(counting_words)
        final_list.append(word_counter_dict)
    
    return final_list
    #print(final_list)

split_and_counter('crude')

# END OF HELPER FUNCTIONS



def part1_load(folder1, folder2, n=1):
    # CHANGE WHATEVER YOU WANT *INSIDE* THIS FUNCTION.
    #return pd.DataFrame(npr.randn(2,2)) # DUMMY RETURN
    
    '''
    This function takes two folders containing text files and transforms them into a list of 
    dictionaries (split_and_count). It also creates a Pandas DataFrame. NaNs are replaced with zeros.

    Args:
        folder1 - path to the first folder
        folder2 - path to the second folder 
        n - int that indicates the max number of times a word can appear to be included (default value is 1).
    Returns:
        DataFrame where there is one column for the filenames, followed by a second one for folder 
        names and an individual column for every word.
    '''

    dict1 = split_and_counter(folder1)
    dict2 = split_and_counter(folder2)

    sum_of_dicts = dict1 + dict2
    df = pd.DataFrame(sum_of_dicts)
    df = df.fillna(0) #replace nan with 0
    df_col = df.columns[2:]
    dropping = []
    for column in df_col:
        val_column = df[column].sum()
        if val_column <= n:
            dropping.append(column)

    dataframe = df.drop(dropping, axis=1)
    return dataframe
    
part1_load('grain' , 'crude', n=5)



def part2_vis(df, m):
    # DO NOT CHANGE
    '''
    The function takes a Panda DataFrame and produces a bar chart with the top m term frequencies
    within the loaded DataFrame.
    
    Args:
        df - DataFrame
        m - top m term frequencies in the DataFrame
    
    Returns: 
        A bar chart where every matching bar represents a different class.
    '''
    
    assert isinstance(df, pd.DataFrame)

    #getting the sum of the columns + sorting them
    sum_col = df.sum()[2:] #sum of columns
    sorted_df = sum_col.sort_values(ascending = False)
    
    #accessing indexes lower than m
    lower_than_m = sorted_df[m:]
    
    #we don't need those columns ranked lower than m because we want
    #a bar chart of the top m term frequencies
    top_m_terms = df.drop(lower_than_m.index, 1)
    
    final_df = top_m_terms.groupby(['classname']).sum().sort_values(top_m_terms['classname'][0], axis=1, ascending = False)
    return final_df.T.plot(kind="bar")


part2_vis(part1_load('grain', 'crude', 20), 10)



def part3_tfidf(df):
    # DO NOT CHANGE
    
    '''
    It takes a DataFrame and creates a new DataFrame where the values are transformed by using tf-idf.
    
    Args:
        df - DataFrame (from part 1)
        
    Returns:
        DataFrame with the values transformed via tf-idf.
    '''
    
    assert isinstance(df, pd.DataFrame)

    dataframe = df.copy()
    for column in dataframe:
        if column != "classname" and column != "filename":  
            #term frequency
            tf = dataframe[column]

            #getting all docs where the term can't be found
            no_term_docs = tf.isin([0]).sum() #.isin() evaluates to True or False
            num_of_docs = len(dataframe)

            #inverse document frequency
            idf = math.log((num_of_docs) / (num_of_docs - no_term_docs))
            dataframe[column] = dataframe[column] * idf
    
    return dataframe

part3_tfidf(part1_load('grain', 'crude', 20))



def part4(df, m):
    # DO NOT CHANGE
    
    '''
    The function takes a Panda DataFrame and produces a bar chart with the top m term frequencies
    within the loaded DataFrame.

    Args:
        df - DataFrame (output of part 3)
        m - top m term frequencies in the DataFrame
        
    Returns: 
        A bar chart where every matching bar represents a different class.
    '''
    
    assert isinstance(df, pd.DataFrame)

    #getting the sum of the columns + sorting them
    sum_col = df.sum()[2:] #sum of columns
    sorted_df = sum_col.sort_values(ascending = False)
    
    #accessing indexes lower than m
    lower_than_m = sorted_df[m:]
    
    #we don't need those columns ranked lower than m because we want
    #a bar chart of the top m term frequencies
    top_m_terms = df.drop(lower_than_m.index, 1)
    
    final_df = top_m_terms.groupby(['classname']).sum().sort_values(top_m_terms['classname'][0], axis=1, ascending = False)
    return final_df.T.plot(kind="bar")

  
part4(part3_tfidf(part1_load('grain', 'crude', 20)), 10)



# ADD WHATEVER YOU NEED HERE, INCLUDING BONUS CODE.

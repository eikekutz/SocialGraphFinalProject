import io 
from os import listdir
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import *
import re 
import string
from langdetect import detect
import langdetect
import pandas
from collections import defaultdict
import pickle
import ast
import os




class TextAnalyser:

    DATA_PATH = 'data/Top50/reviews'

    def writeFile(self, f, fname):
    #The function writes a pickle file storing the element given in input.
        with open('data/Top50/reviews/Pickles/' + fname, 'wb') as handle:
            pickle.dump(f, handle, protocol = pickle.HIGHEST_PROTOCOL)
            
    def readFile(self, fname):
    #The function read a pickle file and return the related structure.
        res = None
        with open(DATA_PATH + '/' + fname, 'rb') as handle:
            res = pickle.load(handle)
        return res

    def CreateFiles_WithEnglishReviewsOnly(self, path = DATA_PATH):
        # go through files in directory and create equivalent ones where the elements where the reviews and response were not in English are 0
        # creates dictionary with the count of languages the reviews (hopefullt this means users) for each city
        # creates dictionary with the response percentage for each user

        self.language_count = {}
        self.response_frequency = defaultdict(int)
        times_user_was_reviewed = defaultdict(int)
        files = [f for f in listdir(path) if os.path.isfile(path+'/'+f) == True]
        for file in files:
            #eliminate Rev.txt part at the end
            CityName = file[:-7]
            self.language_count[CityName] = defaultdict(int)
            CityReviews_DataFrame = pandas.read_csv(path +'/'+file, encoding='utf-8', engine='c')

            # populate dict with users and how many times they were reviewed
            for user, number_reviews in CityReviews_DataFrame['to'].value_counts().items():
                    times_user_was_reviewed[user] += number_reviews

            i = 0
            for index, row in CityReviews_DataFrame.iterrows():
                
                try:
                    id_user_reviewed = row['to']
                    languageOfReview = detect(row['text'])
                    if languageOfReview == 'en':
                        self.language_count[CityName]['en'] += 1
                
                    else:
                        #put review to  0 and add to dictionary this language count
                        CityReviews_DataFrame['text'][index] = 0
                        self.language_count[CityName][languageOfReview] += 1  
                    
                    # not nan
                    if len(str(row['response'])) > 4:
                        self.response_frequency[id_user_reviewed] += 1                 
                    i += 1

                except langdetect.lang_detect_exception.LangDetectException as k: #and pandas.errors.ParserError:
                    pass 
                
                if(i%500 == 0):      
                    print(i,'/',len(CityReviews_DataFrame))
    
            CityReviews_DataFrame.to_csv(path + '/English_Only/' + CityName + "Rev_EnglishOnly.csv")
            print("Done with %s" %(file))

        # divide to make frequency
        for key,value in self.response_frequency.items():
            self.response_frequency[key] = value / times_user_was_reviewed[key]

        self.writeFile(self.language_count, "Language_count_dict.pickle")
        self.writeFile(self.response_frequency, "Response_Frequency_dict.pickle")
        

    #def tester(self, path = DATA_PATH):

    def ParseReviews (self, path = 'data/Top50/reviews/English_Only'):
    #filter review content and outputs data structure
        files =[f for f in os.listdir(path)]

        for file in files:
            CityReviews_DataFrame = pandas.read_csv(path +'/'+file, encoding='utf-8', engine='c')
            for index, row:
            self.frequency_distributions = {}
            StopWords = set(stopwords.words('english'))
            Stemmer = nltk.stem.PorterStemmer()
            tokens = word_tokenize( **input** )
            words = []
            for word in tokens:
                word = word.lower()
                if not bool(re.search("[^A-Za-z]",word)) and word not in StopWords:
                    words.append((stemmer.stem(word))

            
                
        
             


    #def save_file(self,file):

        







        







    

    



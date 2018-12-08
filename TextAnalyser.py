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
import math
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
import matplotlib
from nameparser import HumanName
import numpy as np
import string


class TextAnalyser:
    

    DATA_PATH = 'data/Top50/reviews'
    DATA_PATH_ENGLISH = 'data/Top50/reviews/English_Only'
    DATA_PATH_PICKLES = 'data/Top50/reviews/Pickles'
    DATA_PATH_NAMES = 'data/Top50/reviews/Names'
    citynames = [f[:-19] for f in os.listdir(DATA_PATH_ENGLISH)]
    fileNamesEnglishOnly = [f for f in os.listdir(DATA_PATH_ENGLISH)]

    def __init__(self):

        self.WordFrequencyDistribution_PerCity = {}
        self.invertedIndex = {}
        self.names = []
        #self.FrequencyDistribution_PerUser = {}

    def expandNamesList (self, file, file_path = DATA_PATH_NAMES):

        StopWords = set(stopwords.words('english'))
        names_dataframe = pandas.read_csv(file_path + '/' + file,encoding='ISO-8859-1' )
        i = 0
        for index,row in names_dataframe.iterrows():
            try:
                tokens = word_tokenize(row['names'])
                for name  in tokens:
                    name = name.lower()
                    if not bool(re.search("[^A-Za-z]",name)) and name not in StopWords:
                        self.names.append(name)
                i +=1
                if (i%10000 == 0):
                    print(i)
            except TypeError as t: #UnicodeDecodeError):  
                pass
        self.names = list(set(self.names))
        self.writeFile(self.names, "PeopleNames_List.pickle")
        
    def writeFile(self, f, fname):
    #The function writes a pickle file storing the element given in input.
        with open('data/Top50/reviews/Pickles/' + fname, 'wb') as handle:
            pickle.dump(f, handle, protocol = pickle.HIGHEST_PROTOCOL)
            
    def readFile(self, fname, path = DATA_PATH_PICKLES):
    #The function read a pickle file and return the related structure.
        res = None
        with open(path + '/' + fname, 'rb') as handle:
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
        

   # def tester(self, path = DATA_PATH):

    def GetPositiveNegative_Experiences_PerCity (self, path = DATA_PATH):
        #saves dict with count of type of experiences (positive,negative,neutral per city)
        #saves dict with positive / negative  ratio per city

        self.PositiveNegativeNeutral_Count = defaultdict(list)
        self.NegativePositive_Ratio = defaultdict(float)
        files = [f for f in listdir(path) if os.path.isfile(path+'/'+f) == True]
        for file in files:
            CityName = file[:-7]
            CityReviews_DataFrame = pandas.read_csv(path +'/'+ file, encoding='utf-8', engine='c')
            #experiences can be positive, negative or neutral
            for type_of_experience,  count in CityReviews_DataFrame['experience'].value_counts().items():
                self.PositiveNegativeNeutral_Count[CityName].append((type_of_experience,count))
        for key,value in self.PositiveNegativeNeutral_Count.items():
            self.NegativePositive_Ratio[key] = (value[1][1] / value[0][1]) * 100

        self.writeFile(self.NegativePositive_Ratio, "NegativePositiveRatio_PerCity_dict.pickle")
        self.writeFile(self.PositiveNegativeNeutral_Count, "TypesOfExperiences_Count_PerCity.pickle") 
        
        
 
    def ParseReviews (self, path = DATA_PATH_ENGLISH):
        #filter review content and save/output dictionary with city as key and the words in the reviews there
        #can use it to create dictionary with users and words received in the reviews when necessary
        
        files =[f for f in os.listdir(path)]
        StopWords = set(stopwords.words('english'))
        Stemmer = nltk.stem.PorterStemmer()

        for file in files:
            CityReviews_DataFrame = pandas.read_csv(path +'/'+file, encoding='utf-8', engine='c')
            words = []
            #words_stemmed = []
            CityName = file[:-19]
            i = 0
            for index, row in CityReviews_DataFrame.iterrows():
                tokens = word_tokenize(row['text'])
                for word in tokens:
                    word = word.lower()
                    if not bool(re.search("[^A-Za-z]",word)) and word not in StopWords:
                        words.append(word)
                        #words_stemmed.append(Stemmer.stem(word))
                i += 1
                if (i%500 == 0): 
                    print(i)
            print("Done with %s" %(CityName))
            self.WordFrequencyDistribution_PerCity[CityName] = nltk.FreqDist(words)

        self.writeFile(self.WordFrequencyDistribution_PerCity, "WordFrequencyDistribution_PerCity_dict.pickle")

    def CreateInvertedIndex (self, word_distribution_percity = 'WordFrequencyDistribution_PerCity_dict.pickle'):
    # create data structure (dictionary) to easily access each word and see in which city reviews it appears for calculation of TF-IDF 
        
        if len(self.WordFrequencyDistribution_PerCity) == 0:
            self.WordFrequencyDistribution_PerCity = self.readFile(word_distribution_percity)

        i = 0
        for city,freqDist in self.WordFrequencyDistribution_PerCity.items():
            number_of_words_in_city = sum(freqDist.values())
            for word,tf in freqDist.items():
                try:
                    self.invertedIndex[word][1][city] = tf/number_of_words_in_city

                except KeyError as k:
                    self.invertedIndex[word] = [None, {city:tf/number_of_words_in_city}]

            i += 1
            print('%s Cities done' %(i))
            
        NumberOfCities = 20
        for word, pair in self.invertedIndex.items():
            #base 10
            pair[0] = math.log(NumberOfCities/len(pair[1]),10)
            
        self.writeFile(self.invertedIndex, 'InvertedIndex.pickle')
        print("Inverted Index Saved")

    def GetCityWordclouds(self, path = DATA_PATH_ENGLISH, invertedIndex = 'InvertedIndex.pickle', with_names = False):
    #create a dictionary of words(keys) and TF-IDF(value) of each city and then do a wordcloud according to that

        citynames = [f[:-19] for f in listdir(path)]
        if len (self.invertedIndex) == 0:
            self.invertedIndex = self.readFile(invertedIndex)
        if len (self.names) == 0:
            self.names = self.readFile('PeopleNames_List.pickle')

        for city in citynames:
            cityDictionary = {}
            for word,pair in self.invertedIndex.items():
                if city in pair[1].keys() and word not in self.names:
                    cityDictionary[word] = pair[0] * pair[1][city]
            #print(sorted(cityDictionary.items(), key=lambda kv: kv[1],reverse=True)[:100])
            wordcloud = WordCloud(background_color='white')
            wordcloud.generate_from_frequencies(frequencies=cityDictionary)
            plt.figure()
            plt.imshow(wordcloud, interpolation="bilinear")
            plt.axis("off")
            plt.savefig('wordclouds/' + city + '_noHumanNames')
            
    #def getSpecificUsersWordclouds
    #def getTopHostsWordcloudsFor Reviews Received

    def CalculateAVGCitySentiment(self, Tblob = False, path = DATA_PATH_ENGLISH):
    # takes the previous built dictionary and outputs a dictionary cointaining the average sentiment per city

        if len(self.WordFrequencyDistribution_PerCity) == 0:
            self.WordFrequencyDistribution_PerCity = self.readFile('WordFrequencyDistribution_PerCity_dict.pickle')

        self.AvgCitySentiment = defaultdict(float)
        SentimentDataSet = pandas.read_csv('data/SentimentAnalysis/Data_Set_S1.csv')

        for city,freqdist in self.WordFrequencyDistribution_PerCity.items():
            
            ListOfWordstoAnalyze = []
            #add word the amount of times it appears
            for word,freq in freqdist.items():
                for i in range(freq):
                    ListOfWordstoAnalyze.append(word)
            
            Sentiment = float(0)
            i = 0
            for index, row in SentimentDataSet.iterrows():
                if row['word'] in ListOfWordstoAnalyze:
                    Sentiment += row['happiness_average'] * ListOfWordstoAnalyze.count(row['word'])
                i += 1
                if (i%2000) == 0:
                    print(i)
            
            Sentiment /= len(ListOfWordstoAnalyze)
            self.AvgCitySentiment[city] = Sentiment 
                
            #ListOfWordstoAnalyze.extend([word] * frequency for word,frequency in freqdist.items())
            print(self.AvgCitySentiment)
            print("Done with %s" %(city))

        self.writeFile(self.AvgCitySentiment[city], 'AvgCitySentiment_dict.pickle')


    def CalculateSentimentofEachReview (self, Tblob = False, path = DATA_PATH_ENGLISH):
    # creates dictionary with keys : review id (which we can then link to each specific user) and review sentiment as values

        self.SentimentPerReview = defaultdict(float)
        SentimentDataSet = pandas.read_csv('data/SentimentAnalysis/Data_Set_S1.csv')
        files = [f for f in os.listdir(path)]
        StopWords = set(stopwords.words('english'))
        Punctuation = set(string.punctuation)
        
        
        for file in files:
            CityReviews_DataFrame = pandas.read_csv(path +'/'+file, encoding='utf-8', engine='python')
            print('Beggining %s' %(file))
            i = 0
            for index,row in CityReviews_DataFrame.iterrows():
               
                if row['text'] != '0':
                    temp_text = "".join(word for word in row['text'] if word not in Punctuation)
                    words = []
                    tokens = word_tokenize(temp_text)

                    for word in tokens:
                        word = word.lower()
                        if not bool(re.search("[^A-Za-z]",word)) and word not in StopWords:
                            words.append(word) 


                    Sentiment = float(0)
                    for word in words:
                        if word in list(SentimentDataSet['word']):
                            Sentiment += SentimentDataSet['happiness_average'][np.where(SentimentDataSet["word"] == word)[0][0]] * words.count(word)

                    if len(words) != 0:
                        Sentiment /= len(words)
                        self.SentimentPerReview[row['id']] = Sentiment  
                    else: 
                        self.SentimentPerReview[row['id']] = False  
                         
                else:
                    self.SentimentPerReview[row['id']] = None     
                
                
                i +=1
                if (i%1000 == 0):
                    (print(i))

        self.writeFile(self.SentimentPerReview, 'SentimentPerReview_Dict.pickle')
                                       
        
    def CreateDictionaryforSentimentPerUser (self):
    #takes previously created dictionary and creates a new one that user is the keys, as values we will have another dictionary that has overall average sentiment on the reviews for that user
    # the average sentiment received as a surfer and the average sentiment received as a host

        if len(self.self.SentimentPerReview) == 0:
            self.self.SentimentPerReview = self.readFile('SentimentPerReview_Dict.pickle.pickle')

        Concatinated_Revi
        for user
        





        



        #return sentimentscore


    #def CalculateAVGSentimentPerCity:







    
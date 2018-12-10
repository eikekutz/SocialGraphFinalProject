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
import seaborn as sns


class TextAnalyser:
    

    DATA_PATH = 'data/Top50/reviews'
    DATA_PATH_ENGLISH = 'data/Top50/reviews/English_Only'
    DATA_PATH_PICKLES = 'data/Top50/reviews/Pickles'
    DATA_PATH_NAMES = 'data/Top50/reviews/Names'
    DATA_PATH_HOSTS = 'data/Top50/hosts'
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


    def CalculateSentimentofEachReview (self, path = DATA_PATH_ENGLISH):
    # creates dictionary with keys : review id (which we can then link to each specific user) and review sentiment as value
    #False as a value means the review had no relevant words to do sentiment analysis on, None means no English

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
                    divide_by = 0
                    for word in words:
                        if word in list(SentimentDataSet['word']):
                            Sentiment += SentimentDataSet['happiness_average'][np.where(SentimentDataSet["word"] == word)[0][0]] 
                            divide_by += 1

                    if divide_by != 0:
                   
                        Sentiment /= divide_by
                        self.SentimentPerReview[row['id']] = Sentiment  
                    else:
                        self.SentimentPerReview[row['id']] = False
                                           
                else:
                    self.SentimentPerReview[row['id']] = None     
                
                
                i +=1
                if (i%1000 == 0):
                    (print(i))

        self.writeFile(self.SentimentPerReview, 'SentimentPerReview_Dict.pickle')
                                       
    def Concact(self, path = DATA_PATH_HOSTS):
    #concatinate new english files
        files = [f for f in listdir(path)]
        rev = pandas.DataFrame() #creates a new dataframe that's empty
        for file in  files:
            df = pandas.read_csv(path +'/'+ file ,index_col=False)
            rev = pandas.concat([rev,df]).reset_index(drop=True)

        rev=rev.drop_duplicates()
        rev.to_csv(path + '/ConcactinatedHosts.csv',index = False)

    def CreateDictionaryforSentimentPerUser (self):
    #takes previously created dictionary and creates a new one that user is the keys, as values we will have another dictionary that has overall average sentiment on the reviews for that user
    # the average sentiment received as a surfer and the average sentiment received as a host

        
        SentimentPerReview = self.readFile('SentimentPerReview_Dict.pickle')

        SentimentPerUser = {}
        ValueDictionary = defaultdict(float)
        ReviewsOfEachUser = defaultdict(list)
        ReviewsOfEachUser_asHost = defaultdict(list)
        ReviewsOfEachUser_asSurfer = defaultdict(list)
        ConcatinatedReviews_DataFrame = pandas.read_csv('data/Top50/reviews/English_Only/EnglishConcact.csv')
        TimesUserWasReviewed = defaultdict(float)
        TimesUserWasReviewed_asHost = defaultdict(float)
        TimesUserWasReviewed_asSurfer = defaultdict(float)
        
        #get review ID's of each user
        #get amount of times user was reviewed both in general and as a surfer and as a host
        for index, row in ConcatinatedReviews_DataFrame.iterrows():
            if row['text'] != '0':
                ReviewsOfEachUser[row['to']].append(row['id'])
                TimesUserWasReviewed[row['to']] += 1
                if row['relationshipType'] == 'surf':
                    TimesUserWasReviewed_asHost[row['to']] += 1
                    ReviewsOfEachUser_asHost[row['to']].append(row['id'])
                    
                elif row['relationshipType'] == 'host':
                    TimesUserWasReviewed_asSurfer[row['to']] += 1
                    ReviewsOfEachUser_asSurfer[row['to']].append(row['id'])

        for user in ReviewsOfEachUser.keys():
            UserValueDictionary = defaultdict(float)
            for review in ReviewsOfEachUser[user]:
                UserValueDictionary['overall'] += SentimentPerReview[review]
                if review in ReviewsOfEachUser_asHost[user]:
                    UserValueDictionary['as_Host'] += SentimentPerReview[review]
                elif review in ReviewsOfEachUser_asSurfer[user]:
                    UserValueDictionary['as_Surfer'] += SentimentPerReview[review]

            UserValueDictionary['overall'] /= TimesUserWasReviewed[user]
            if  UserValueDictionary['as_Host'] > 0:
                UserValueDictionary['as_Host'] /= TimesUserWasReviewed_asHost[user]
            if UserValueDictionary['as_Surfer'] > 0:
                UserValueDictionary['as_Surfer'] /= TimesUserWasReviewed_asSurfer[user] 
                
            SentimentPerUser[user] = UserValueDictionary

        self.writeFile(SentimentPerUser, 'SentimentPerUser.pickle')

    def plotOverallHostSurferGraph_WithVariations_BasedonSentiment (self, dict ='SentimentPerUser.pickle'):

        SentimentPerUser = self.readFile(dict)
        OverallSentiment = []
        SurfingSentiment = []
        HostingSentiment = []

        for user in SentimentPerUser.keys():

            OverallSentiment.append(SentimentPerUser[user]['overall'])
            SurfingSentiment.append(SentimentPerUser[user]['as_Surfer'])
            HostingSentiment.append(SentimentPerUser[user]['as_Host'])
         
        print(sorted(SurfingSentiment[:20]))
        #plt.figure()
        #plt.boxplot(x= [OverallSentiment,SurfingSentiment,HostingSentiment], sym =)
        #plt.savefig('plots/SentimentDistribution')      
        #but I also want to see the correlation between surf reviewns and host reviews

            
    def plotOnlyHostsandSurfers (self, pickle = 'SentimentPerUser.pickle', path = DATA_PATH_HOSTS + '/ConcactinatedHosts.csv'):
    #since we only have info on the hosts of each city we can only plot, percentage wise, the members (hosts) that we have information on
    # within europe(the cities we consider) do they tend to be hosts only surfs only or both
        
        SentimentPerUser = self.readFile(pickle)
        ConcactinatedDataFrame = pandas.read_csv(path)

        Host_0_count = 0
        Surf_0_count = 0
        users = 0
        values = []
       # files = [f for f in os.listdir(path)]
        cities = []
        screwUps = 0
        isthere = 0
        notThere = 0

        for index,row in ConcactinatedDataFrame.iterrows():
            print(type(int(row['id'])))

        #for user in SentimentPerUser.keys():
         #   if int(user) in list(ConcactinatedDataFrame['id']):
          #      
           #         if str(user) == str(ConcactinatedDataFrame['id'][np.where(ConcactinatedDataFrame['id'] == user)[0][0]]):
                   #     isthere += 1

           #     except IndexError as k:
            #        notThere += 1
        #print(notThere, isthere)
                #isthere += 1
                #if SentimentPerUser[user]['as_Host'] == 0:
                 #   screwUps += 1
       # print(screwUps,isthere)










        #for file in files:
            #City__Host_0_count = 0
            #City_Surf_0_count = 0
            #Number_of_

            #CityReviews_DataFrame = pandas.read_csv(file)
            #CityName = file[:-19]
            #cities.append(CityName)
           # for index, row in CityReviews_DataFrame.iterrows():
                #if row['to'] in SentimentPerUser.keys():
                    #print("N")
            #       if SentimentPerUser[row['to']]



            
        #for user in SentimentPerUser.keys():
            #users += 1
            #if SentimentPerUser[user]['overall'] == 0:
             #   Error +=1
            #    print(SentimentPerUser[user])
           # if SentimentPerUser[user]['as_Surfer'] == 0:
           #     Surf_0_count += 1
          #  if SentimentPerUser[user]['as_Host'] == 0:
         #       Host_0_count += 1

        #users -= (Surf_0_count + Host_0_count)    
   
        #ind = [x for x, _ in enumerate(countries)]

        #plt.bar(1,users, width=0.5, label='Users who Both Surf and Host', color='gold', bottom=Surf_0_count + Host_0_count)
        #plt.bar(1,Surf_0_count, width=0.5, label='Only Hosts', color='silver', bottom=Host_0_count)
        #plt.bar(1,Host_0_count, width=0.5, label='Only Surfers', color='#CD853F')
        #plt.xticks(range(1))
        #plt.ylabel("Number of Users")
        #plt.xlabel("Cities")
        #plt.legend(loc="upper right")
        #plt.title("2012 Olympics Top Scorers")

       # plt.savefig("plots/test")

        

        

   #for city and in general

   # def plotEvolutionOfSentimentOvertime:

   #seaplot: visualize linear relationships

   #def plotWordcloudsOfReviewsWithBestAndWorseSentiemtn or maybe just show the reviews?

   # def plotHeatMapofSimilarityBetweenCitiesTf-IDF

    #plot Average Sentiment per city







    def test (self, path = DATA_PATH): 
        
        self.SentimentPerReview = self.readFile('SentimentPerReview_Dict.pickle')
        ConcatinatedReviews_DataFrame = pandas.read_csv('data/Top50/concatinated_reviews.csv')
        SentimentPerReview_Copy = defaultdict(float)

        for review, sentiment in self.SentimentPerReview.items():
            if sentiment != None and sentiment != False:
               SentimentPerReview_Copy[review] = sentiment
            

        print(sorted(SentimentPerReview_Copy.items(), key = lambda kv: kv[1], reverse = False)[0:50])

    def test1 (self, path = DATA_PATH): 

        dictionario = self.readFile('Language_count_dict.pickle')
        print(dictionario)

    def test2 (self, file = 'data/reviews_total_geo_2_test.csv'):

        df = pandas.read_csv(file)
        StopWords = set(stopwords.words('english'))
        Punctuation = set(string.punctuation)    
        SentimentDataSet = pandas.read_csv('data/SentimentAnalysis/Data_Set_S1.csv')
        #print(df)
        for index,row in df.iterrows():
                    
            if row['text'] != '0':
                    words = []
                    tokens = word_tokenize(row['text'])

                    for word in tokens:
                        word = word.lower()
                        if not bool(re.search("[^A-Za-z]",word)) and word not in StopWords:
                            words.append(word)
                            
                    #print(words)
                    #print(len(words)) 


                    Sentiment = float(0)
                    divide_by = 0
                    for word in words:
                        if word in list(SentimentDataSet['word']):
                            #print(word)
                            Sentiment += SentimentDataSet['happiness_average'][np.where(SentimentDataSet["wnpord"] == word)[0][0]] 
                            divide_by += 1
                    print(Sentiment)

              
                    Sentiment /= divide_by

                    print(Sentiment)
             
    def test_SentimentPerUser (self, pickle = 'SentimentPerUser.pickle'):    

        SentimentPerUser = self.readFile(pickle)
        Host_0_count = 0
        Surf_0_count = 0
        Error = 0

        for user in SentimentPerUser.keys():
            
            if SentimentPerUser[user]['overall'] == 0:
                Error +=1
                print(SentimentPerUser[user])
            if SentimentPerUser[user]['as_Surfer'] == 0:
                Surf_0_count += 1
            if SentimentPerUser[user]['as_Host'] == 0:
                Host_0_count += 1

        print(Host_0_count,Surf_0_count,Error)
        print(len(SentimentPerUser.keys()))




                
                
               
            









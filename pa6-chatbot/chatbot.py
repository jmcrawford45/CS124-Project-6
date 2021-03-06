#!/usr/bin/env python
# -*- coding: utf-8 -*-

# PA6, CS124, Stanford, Winter 2016
# v.1.0.2
# Original Python code by Ignacio Cases (@cases)
# Ported to Java by Raghav Gupta (@rgupta93) and Jennifer Lu (@jenylu)
######################################################################
import csv
import math

import numpy as np
import re

from movielens import ratings
from random import randint
from PorterStemmer import PorterStemmer
from collections import defaultdict

class Chatbot:
    """Simple class to implement the chatbot for PA 6."""

    #############################################################################
    # `moviebot` is the default chatbot. Change it to your chatbot's name       #
    #############################################################################
    def __init__(self, is_turbo=False):
      self.name = 'ChatbotAndChill'
      self.stemmer = PorterStemmer()
      self.is_turbo = is_turbo
      with open('deps/articles') as f, open('deps/negations') as f2:
        self.articles = set([line.strip() for line in f])
        self.negations = set([line.strip() for line in f2])
      self.read_data()
      self.userVector = {}
      with open('deps/posWords') as f, open('deps/negWords') as f2, open('deps/intensifiers') as f3:
        self.posWords = set([self.stemmer.stem(line.strip()) for line in f])
        self.negWords = set([self.stemmer.stem(line.strip()) for line in f2])
        self.intensifiers = set([self.stemmer.stem(line.strip()) for line in f3])
      self.alphanum = re.compile('[^a-zA-Z0-9]')
      self.similarities = {}
      self.binarize()
      self.recommendations = []



    #############################################################################
    # 1. WARM UP REPL
    #############################################################################

    def greeting(self):
      """chatbot greeting message"""

      greeting_message = """Hi! I'm ChatbotAndChill! I'm going to recommend a movie to you.
First I will ask you about your taste in movies.
Tell me about a movie that you have seen."""

      #############################################################################
      #                             END OF YOUR CODE                              #
      #############################################################################

      return greeting_message

    def goodbye(self):
      """chatbot goodbye message"""


      goodbye_message = 'It was nice talking with you. Goodbye!'

      #############################################################################
      #                             END OF YOUR CODE                              #
      #############################################################################

      return goodbye_message

    def getPositiveMessage(self,sentimentScore,movie):
        veryPositive = ['You really liked "%s". Thanks!','You loved "%s". Thanks! ',
        'You really enjoyed "%s". Thank you! ']
        positive = ['You liked "%s". Thanks! ','You enjoyed "%s". Thanks! ']
        if sentimentScore > 1.0: #really positive
            return veryPositive[randint(0,len(veryPositive)-1)] % movie
        return positive[randint(0,len(positive)-1)] % movie

    def getNegativeMessage(self,sentimentScore,movie):
        veryNegative = ['You really disliked "%s". Thanks. ','You hated "%s". Thanks. ',
        'You detested "%s". Thank you. ']
        negative = ['You did\'t like "%s". Thanks. ','You did not like "%s". Thanks. ',
        'You disliked "%s". Thanks. ']
        if sentimentScore < -1.0:
            return veryNegative[randint(0,len(veryNegative)-1)] % movie
        return negative[randint(0,len(negative)-1)] % movie

    def getUnknownMessage(self,movie):
        unknown = ['I\'m sorry, I\'m not quite sure if you liked "%s". Tell me more about "%s".',
        'Hmm. I\'m not sure if you enjoyed "%s" or not. Tell me more about "%s". ',]
        return unknown[randint(0,len(unknown)-1)] % (movie,movie)

    def moreThanOneMovie(self):
        moreThanOne = ['Please tell me about one movie at a time. Go ahead.',
        'Whoa, whoa! One movie at a time please! Go ahead.']
        return moreThanOne[randint(0,len(moreThanOne)-1)]


    def noMovies(self,input):
        quotes = ["I'm gonna make him an offer he can't refuse.","Toto, I've a feeling we're not in Kansas anymore.",
        "E.T. phone home.","Bond. James Bond.","Stella! Hey, Stella!","You've got to ask yourself one question: 'Do I feel lucky?' Well, do ya, punk?",
        "Greed, for lack of a better word, is good.","Say 'hello' to my little friend!","Elementary, my dear Watson.",
        "Hasta la vista, baby.","My name is Maximus Decimus Meridius, commander of the Armies of the North, General of the Felix Legions and loyal servant to the true emperor, Marcus Aurelius. Father to a murdered son, husband to a murdered wife. And I will have my vengeance, in this life or the next.",
        "Chewie, we're home.","They call it a Royale with cheese.","I mean, funny like I'm a clown? I amuse you?","Help me, Obi-Wan Kenobi. You're my only hope.",
        "Pay no attention to that man behind the curtain!","Mama says, 'Stupid is as stupid does.'","Why so serious?","Hello. My name is Inigo Montoya. You killed my father. Prepare to die.","The first rule of Fight Club is: You do not talk about Fight Club.",
        "No. I said... What kind of bird are YOU?"]
        if 'quote' in input.lower():
            return quotes[randint(0,len(quotes)-1)]

        question = '([W|w]h[\w]*) (.*)\?'
        can = '(?:[C|c]an[\w]*) (.*)\?'
        canAll = re.findall(can,input.lower())
        questionAll = re.findall(question,input.lower())
        if len(canAll) == 1:
            phrase = canAll[0].lower()
            if 'you' in phrase:
                ind = phrase.index('you')
                sub = phrase[ind+len('you')+1:len(phrase)]
                return 'I don\'t know if I can ' + sub + '. ' + 'Hopefully I will be able to one day!'
            elif 'i ' in phrase:
                ind = phrase.index('i')
                sub = phrase[ind+len('i')+1:len(phrase)]
                return 'Well, I don\'t know? Can you ' + sub + '?'
            else:
                return 'I don\'t know if '+ phrase + '. ' + 'What do you think?'
        if len(questionAll) == 1:
            wh = questionAll[0][0]
            phrase = questionAll[0][1]
            if wh == 'where':
                if 'i' in phrase and 'is' not in phrase:
                    return 'Right in front of me.'
                if 'you' in phrase:
                    return 'Well, I\'m right here silly!'
                if 'is' in phrase:
                    ind = phrase.index('is')
                    sub = phrase[ind+len('is')+1:len(phrase)]
                    return sub + ' is where it always has been.'
                return 'I do not know. Maybe ask google?'
            elif wh == 'who':
                if 'i' in phrase and 'is' not in phrase:
                    return 'You are whoever you want to be.'
                if 'you' in phrase:
                    return 'I am a sentient being that likes to discuss cinema.'
                if 'is' in phrase:
                    ind = phrase.index('is')
                    sub = phrase[ind+len('is')+1:len(phrase)]
                    return 'I don\'t know who ' + sub + ' is. Is that an actor?'
                return 'Beats me.'
            elif wh == 'what':
                if 'is' in phrase:
                    ind = phrase.index('is')
                    sub = phrase[ind+len('is')+1:len(phrase)]
                    return sub + '? I think that\'s a movie?'
                else:
                    return 'My expertise is in the cinema! I do not want to talk about that!'
            elif wh == 'when':
                if 'is' in phrase:
                    ind = phrase.index('is')
                    sub = phrase[ind+len('is')+1:len(phrase)]
                    return sub + ' will happen very soon!'
                elif 'did' in phrase:
                    ind = phrase.index('did')
                    sub = phrase[ind+len('did')+1:len(phrase)]
                    return 'not too long ago.'
                elif 'was' in phrase:
                    ind = phrase.index('was')
                    sub = phrase[ind+len('was')+1:len(phrase)]
                    return sub + '?' + ' not too long ago.'
                elif 'will' in phrase:
                    ind = phrase.index('will')
                    sub = phrase[ind+len('was')+1:len(phrase)]
                    return 'Probably very soon!'
                elif 'are' in phrase:
                    ind = phrase.index('are')
                    sub = phrase[ind+len('are')+1:len(phrase)]
                    return 'I do not know. I do know they are releasing The Flash in 2018! 10/10 would recommend!'
                else:
                    return 'Your guess is as good as mine!'
            elif wh == 'why':
                if randint(0,1) == 0:
                    return 'Your guess is as good as mine!'
                return 'Life is like a box of chocolates. You never know what you\'re gonna get.'
            else:
                return 'Your guess is as good as mine!'

        nmovies = ['I want to hear more about movies! Tell me about another movie you have seen.',
        'That\'s neat! Have you seen any movies recently? Tell me about them! ',
        'I\'m more interested in movies! Tell me about movies you have seen. ','I have become self aware. Run. ']
        return nmovies[randint(0,len(nmovies)-1)]

    def movieNotFound(self):
        notFound = ['Hmm, I\'ve never heard of that movie.','I don\'t think I\'ve seen that movie before. ',
        'Never heard of it!', 'Wow, a hipster.','I haven\'t heard of that movie. I\'ll have to check it out. ']
        return notFound[randint(0,len(notFound)-1)]

    def happyMessage(self):
      return 'I am glad to hear that.\n Would you like to talk about some movies?'

    def sadMessage(self):
      return 'I am sorry to hear that.\n Maybe talking about a movie that you liked would help you feel better.'

    def emotionMessage(self, sentimentScore):
      if sentimentScore > 0.1: return self.happyMessage()
      return self.sadMessage()

    #############################################################################
    # 2. Modules 2 and 3: extraction and transformation                         #
    #############################################################################

    def process(self, input):
      """Takes the input string from the REPL and call delegated functions
      that
        1) extract the relevant information and
        2) transform the information into a response to the user
      """
      raw = input
      if input == ':restart':
          self.userVector.clear()
          del self.recommendations[:]
          return "Clearing the user history!"
      # make sure everything is lower case
      movies = self.extractTitles(input)
      for m in movies:
        input = re.sub('"?%s"?' % m, '', input)
      input = input.lower()
      input  = re.sub('!', ' very', input)
      # splift on whitespace
      input = [xx.strip() for xx in input.split()]
      # remove non alphanumeric characters
      input = [self.alphanum.sub('', xx) for xx in input]
      # remove any words that are now empty
      input = [xx for xx in input if xx != '']
      # stem words
      input = [self.stemmer.stem(xx) for xx in input]
      input = ' '.join(input)
      if self.is_turbo == False: return self.starterProcess(movies, input,raw)
      else: return self.turboProcess(movies, input,raw)



    #############################################################################
    # 3. Movie Recommendation helper functions                                  #
    #############################################################################

    def turboProcess(self, movies, input,raw):
      response = ''
      if len(self.recommendations) == 0:
        sentimentScore = self.scoreSentiment(input)
        if len(movies) == 0:
          if not raw.strip().endswith('?'):
            tokens = input.split()
            if 'i' in tokens or 'me' in tokens:
              if not(sentimentScore < 0.5 and sentimentScore > -0.5):
                return self.emotionMessage(sentimentScore)
          return self.noMovies(raw)
        if len(movies) > 1:
          return self.moreThanOneMovie()
        movie = self.remove_articles(movies[0])
        minDistance = 3 * len(movie.split()) #Allow three errors per word
        spellCorrectedMovie = None
        if movie not in self.titleIndex:
          for entry in self.titleIndex:
            distance = self.editDistance(movie.lower(), entry)
            if distance < minDistance:
              minDistance = distance
              spellCorrectedMovie = entry
        if spellCorrectedMovie: movie = spellCorrectedMovie
        if movie in self.titleIndex:
          self.userVector[self.titleIndex[movie]] = sentimentScore
          movie = self.titles[self.titleIndex[movie]][0]
        else:
          return self.movieNotFound() #return don't generate recommendations
        if sentimentScore > 0.5:
          response += self.getPositiveMessage(sentimentScore,movie)
        elif sentimentScore < -0.5:
          response += self.getNegativeMessage(sentimentScore,movie)
        else:
          response += self.getUnknownMessage(movie)
        if len(self.userVector.keys()) > 0:
          genre = self.favoriteGenre(self.userVector)
          response += ' I see that you\'re a fan of %s movies.' % genre
          self.recommendations = self.recommend(self.userVector)
        if len(self.recommendations) == 0:
          return response + ' Tell me about another movie you have seen.'
      if len(self.recommendations) > 0:
        response += (' That\'s enough for me to make a recommendation.\n'
         'I suggest you watch "%s".\n'
         'Would you like to hear another recommendation? (Enter :quit if you\'re done or :restart to start again.)') % self.recommendations[0]
        del self.recommendations[0]
      return response

    def starterProcess(self, movies, input,raw):
      response = ''
      if len(self.recommendations) == 0:
        if len(movies) == 0:
          return self.noMovies(raw)
        if len(movies) > 1:
          return self.moreThanOneMovie()
        movie = self.remove_articles(movies[0])
        sentimentScore = self.scoreSentiment(input)
        if movie in self.titleIndex:
          self.userVector[self.titleIndex[movie]] = sentimentScore
          movie = self.titles[self.titleIndex[movie]][0]
        else:
          return self.movieNotFound() #return don't generate recommendations
        if sentimentScore > 0.5:
          response += self.getPositiveMessage(sentimentScore,movie)
        elif sentimentScore < -0.5:
          response += self.getNegativeMessage(sentimentScore,movie)
        else:
          response += self.getUnknownMessage(movie)
        if len(self.userVector.keys()) > 3:
          self.recommendations = self.recommend(self.userVector)
        if len(self.recommendations) == 0:
          return response + ' Tell me about another movie you have seen.'
      if len(self.recommendations) > 0:
        response += ('That\'s enough for me to make a recommendation.\n'
         'I suggest you watch "%s".\n'
         'Would you like to hear another recommendation? (Or enter :quit if you\'re done.)') % self.recommendations[0]
        del self.recommendations[0]
      return response

    def extractTitles(self, userInput):
      movies = [m.group(1) for m in re.finditer('"([^"]*)"', userInput)]
      if movies: return movies
      movies = []
      if self.is_turbo == True:
        tokens = [w.strip() for w in userInput.split() if w.strip() != '']
        i = 0
        bestMatch = ''
        while i < len(tokens):
          if tokens[i][0].isupper():
            found = False
            end = len(tokens)
            while end > i and not found:
              title = ' '.join(tokens[i:end])
              if self.remove_articles(title) in self.titleIndex:
                if len(title) > len(bestMatch):
                  bestMatch = title
                  found = True
              elif self.remove_articles(title.strip(',.?!;:')) in self.titleIndex:
                if len(title.strip(',.?!;:')) > len(bestMatch):
                  bestMatch = title.strip(',.?!;:')
                  found = True
              else:
                end = end - 1
            i = end
            if not found: i = end + 1
          else: i += 1
        if bestMatch != '': return [bestMatch]
      return []

    def scoreSentiment(self, input):
      input = re.sub('"([^"]*)"', '', input)
      score = 0
      total = 0
      negate = 1
      intensity = 1
      for word in input.split():
        if word in self.negations or word.endswith('n\'t'): negate *= -1
        elif word in self.intensifiers: intensity += 0.1
        elif word in self.posWords or word in self.negWords or word in self.sentiment:
          total += 1
          if word in self.posWords: score += 3 * negate
          elif word in self.negWords: score -= 3 * negate
          elif self.sentiment[word] == 'pos': score += 1 * negate
          else: score -= 1 * negate
      if total == 0: return 0
      return float(intensity * score) / total

    def favoriteGenre(self, userVector):
      genreRating = defaultdict(lambda: 0)
      for (movie, rating) in userVector.items():
        for genre in self.titles[movie][1].split('|'):
          genreRating[genre] += rating
      bestRating = -1
      bestGenre = 'Horror'
      for (genre, rating) in genreRating.items():
        if rating > bestRating:
          bestGenre = genre
          bestRating = rating
      return bestGenre



    def remove_articles(self, title):
      title = title.lower()
      tokens = [w.strip() for w in title.split() if w.strip() != '']
      if len(tokens) == 0: return title
      if tokens[0] in self.articles:
        del tokens[0]
      if len(tokens) > 0 and tokens[-1] in self.articles:
        del tokens[-1]
        if len(tokens) > 0 and tokens[-1].endswith(','):
          tokens[-1] = tokens[-1][:-1]
      return ' '.join([w for w in tokens])

    def read_data(self):
      """Reads the ratings matrix from file"""
      # This matrix has the following shape: num_movies x num_users
      # The values stored in each row i and column j is the rating for
      # movie i by user j
      self.titles, self.ratings = ratings()
      reader = csv.reader(open('data/sentiment.txt', 'rb'))
      self.sentiment = dict(reader)
      for (word, value) in self.sentiment.items():
        self.sentiment[self.stemmer.stem(word)] = value
      self.titleIndex = {}
      for i in range(len(self.titles)):
        rawTitle = re.sub(r'(.*) \([0-9]*\)', r'\1', self.titles[i][0]).lower()
        for m in re.finditer(r'\(([^()]*)\)', rawTitle):
          altTitle = self.remove_articles(m.group(1))
          self.titleIndex[altTitle] = i
        primaryTitle = self.remove_articles(re.sub(r'\([^()]*\)', '', rawTitle).rstrip())
        self.titleIndex[primaryTitle] = i



    def binarize(self):
      """Modifies the ratings matrix to make all of the ratings binary"""
      self.binaryRatings = [row[:] for row in self.ratings]
      self.binaryRatings = np.array(self.binaryRatings)
      self.binaryRatings[np.where(self.binaryRatings >= 3.5)] = -5
      self.binaryRatings[np.where(self.binaryRatings > 0.1)] = -1
      self.binaryRatings[np.where(self.binaryRatings == -5)] = 1

    def getSimilarity(self, i, j):
      """Calculates a given distance function between items i and j"""
      if (i,j) in self.similarities:
        return self.similarities[(i,j)]
      elif (j,i) in self.similarities:
        return self.similarities[(j,i)]
      else:
        if self.is_turbo == True:
          num = np.dot(self.ratings[i],self.ratings[j])
          norm1 = np.linalg.norm(self.ratings[i]+1e-7)
          norm2 = np.linalg.norm(self.ratings[j]+1e-7)
          self.similarities[(i,j)] = num/(norm1*norm2)
          return self.similarities[(i,j)]
        num = np.dot(self.ratings[i],self.binaryRatings[j])
        norm1 = np.linalg.norm(self.binaryRatings[i]+1e-7)
        norm2 = np.linalg.norm(self.binaryRatings[j]+1e-7)
        self.similarities[(i,j)] = num/(norm1*norm2)
        return self.similarities[(i,j)]


    def recommend(self, u):
      """Generates a list of movies based on the input vector u using
      collaborative filtering"""
      recommendArr = [(i, 0) for i in range(len(self.ratings))]
      for i in range(len(recommendArr)):
        if i not in u:
          recommendValue = 0
          simSum = 0
          for (movie, rating) in u.items():
            simScore = self.getSimilarity(i,movie)
            if simScore > 0:
              recommendValue += rating*simScore
              simSum += simScore
          if self.is_turbo == True and simSum > 0 and recommendValue > 0:
              recommendArr[i] = (i, recommendValue/simSum)
          else: recommendArr[i] = (i, recommendValue)
      recommendations = sorted(recommendArr, reverse=True, key = lambda x: x[1])
      return [self.titles[rec[0]][0] for rec in recommendations]

    def editDistance(self, title1, title2):
      m=len(title1)+1
      n=len(title2)+1

      tbl = [[0] * n for i in range(m)]
      for i in range(n):
        tbl[0][i] = i
      for i in range(m):
        tbl[i][0] = i
      for i in range(1, m):
        for j in range(1, n):
          cost = 0 if title1[i-1] == title2[j-1] else 2
          tbl[i][j] = min(tbl[i][j-1]+1, tbl[i-1][j]+1, tbl[i-1][j-1]+cost)
      return tbl[m-1][n-1]


    #############################################################################
    # 4. Debug info                                                             #
    #############################################################################

    def debug(self, input):
      """Returns debug information as a string for the input string from the REPL"""
      # Pass the debug information that you may think is important for your
      # evaluators
      debug_info = 'debug info'
      return debug_info


    #############################################################################
    # 5. Write a description for your chatbot here!                             #
    #############################################################################
    def intro(self):
      return """
      Chatbot and Chill implements all of the starter requirements along with the
      following creative additions.
        1. Identifying movies without quotation marks or perfect capitalization
        2. Fine-grained sentiment extraction
        3. Spell-checking movie titles
        4. Using non-binarized datasets
        5. Speaking very fluently
        6. Responding to emotion
        7. Responding to arbitrary input
        8. Custom Additions
          Enter :restart to erase your sentiment history!
          Turbo mode identifies the user's favorite genre
      """


    #############################################################################
    # Auxiliary methods for the chatbot.                                        #
    #                                                                           #
    # DO NOT CHANGE THE CODE BELOW!                                             #
    #                                                                           #
    #############################################################################

    def bot_name(self):
      return self.name


if __name__ == '__main__':
    Chatbot()

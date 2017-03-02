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

class Chatbot:
    """Simple class to implement the chatbot for PA 6."""

    #############################################################################
    # `moviebot` is the default chatbot. Change it to your chatbot's name       #
    #############################################################################
    def __init__(self, is_turbo=False):
      self.name = 'moviebot'
      self.is_turbo = is_turbo
      with open('deps/articles') as f, open('deps/negations') as f2:
        self.articles = set([line.strip() for line in f])
        self.negations = set([line.strip() for line in f2])
      self.read_data()
      self.userVector = [0] * len(self.ratings)
      self.stemmer = PorterStemmer()
      with open('deps/posWords') as f, open('deps/negWords') as f2, open('deps/intensifiers') as f3:
        self.posWords = set([self.stemmer.stem(line.strip()) for line in f])
        self.negWords = set([self.stemmer.stem(line.strip()) for line in f2])
        self.intensifiers = set([self.stemmer.stem(line.strip()) for line in f3])
      self.alphanum = re.compile('[^a-zA-Z0-9]')



    #############################################################################
    # 1. WARM UP REPL
    #############################################################################

    def greeting(self):
      """chatbot greeting message"""
      #############################################################################
      # TODO: Write a short greeting message                                      #
      #############################################################################

      greeting_message = 'How can I help you?'

      #############################################################################
      #                             END OF YOUR CODE                              #
      #############################################################################

      return greeting_message

    def goodbye(self):
      """chatbot goodbye message"""
      #############################################################################
      # TODO: Write a short farewell message                                      #
      #############################################################################

      goodbye_message = 'Have a nice day!'

      #############################################################################
      #                             END OF YOUR CODE                              #
      #############################################################################

      return goodbye_message


    #############################################################################
    # 2. Modules 2 and 3: extraction and transformation                         #
    #############################################################################

    def process(self, input):
      """Takes the input string from the REPL and call delegated functions
      that
        1) extract the relevant information and
        2) transform the information into a response to the user
      """
      #############################################################################
      # TODO: Implement the extraction and transformation in this method, possibly#
      # calling other functions. Although modular code is not graded, it is       #
      # highly recommended                                                        #
      #############################################################################
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
      response = ''
      if self.is_turbo == False:
        if len(movies) == 0:
          return 'I want to hear more about movies! Tell me about another movie you have seen.'
        if len(movies) > 1:
          return 'Please tell me about one movie at a time. Go ahead.'
        m = movies[0]
        movie = self.remove_articles(m)
        sentimentScore = self.scoreSentiment(input)
        if sentimentScore > 0.5:
          response += 'You liked "%s". Thank you!' % movie
        elif sentimentScore < -0.5:
          response += 'You did not like "%s". Thank you!' % movie
        else:
          response += 'I\'m sorry, I\'m not quite sure if you liked "%s". Tell me more about "%s".' % (movie, movie)
        if movie in self.titleIndex:
          self.userVector[self.titleIndex[movie]] = sentimentScore
        else:
          response = 'Hmm, I\'ve never heard of that movie.'
        #print(self.recommend(self.userVector)[:3])
        return response + ' Tell me about another movie you have seen.'


    #############################################################################
    # 3. Movie Recommendation helper functions                                  #
    #############################################################################

    
    def extractTitles(self, userInput):
      movies = [m.group(1) for m in re.finditer('"([^"]*)"', userInput)]
      if movies: return movies
      movies = []
      if self.is_turbo:
        tokens = [w.strip() for w in userInput.split() if w.strip() != '']
        i = 0
        bestMatch = ''
        while i < len(tokens):
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
      for i in range(len(self.ratings)):
        movie = self.ratings[i]
        for j in range(len(movie)):
          rating = movie[j]
          if rating > 0 and rating < 3.5:
            self.binaryRatings[i][j] = -1
          elif rating > 0:
            self.binaryRatings[i][j] = 1
          else:
            self.binaryRatings[i][j] = 0


    def distance(self, u, v):
      """Calculates a given distance function between vectors u and v"""
      # TODO: Implement the distance function between vectors u and v]
      # Note: you can also think of this as computing a similarity measure
      dotProduct = 0
      for a,b in zip(u,v):
        dotProduct += a * b
      return dotProduct

    def recommend(self, u):
      """Generates a list of movies based on the input vector u using
      collaborative filtering"""
      # TODO: Implement a recommendation function that takes a user vector u
      # and outputs a list of movies recommended by the chatbot
      bestSimilarity = -1
      bestUser = 0
      for i in range(len(self.ratings)):
        user = self.ratings[i]
        similarity = self.distance(u, user)
        if similarity > bestSimilarity:
          bestSimilarity = similarity
          bestUser = i
      recommendations = [self.ratings[bestUser] for i in range(len(u)) if u[i] == 0]
      recommendations = sorted(recommendations, reverse=True)
      return recommendations


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
      Your task is to implement the chatbot as detailed in the PA6 instructions.
      Remember: in the starter mode, movie names will come in quotation marks and
      expressions of sentiment will be simple!
      Write here the description for your own chatbot!
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

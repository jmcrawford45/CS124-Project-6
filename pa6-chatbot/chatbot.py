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

class Chatbot:
    """Simple class to implement the chatbot for PA 6."""

    #############################################################################
    # `moviebot` is the default chatbot. Change it to your chatbot's name       #
    #############################################################################
    def __init__(self, is_turbo=False):
      self.name = 'moviebot'
      self.is_turbo = is_turbo
      self.read_data()
      self.userVector = [0] * len(self.ratings[0])

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
      if self.is_turbo == True:
        response = 'processed %s in creative mode!!' % input
      else:
        response = 'processed %s in starter mode' % input
      for m in re.finditer('"([^"]*)"', input):
        movie = m.group(1)
        response += '\nDiscovered movie: %s' % movie
        sentimentScore = self.scoreSentiment(input)
        if movie in self.titleIndex:
          self.userVector[self.titleIndex[movie]] = sentimentScore
          response += '\nMovie preference added to vector'
        print(self.recommend(self.userVector)[:3])
        if sentimentScore > 0.5:
          response += '\nYou liked "%s". Thank you!' % movie
        elif sentimentScore < -0.5:
          response += '\nYou did not like "%s". Thank you!' % movie
        else:
          response += 'I\'m sorry, I\'m not quite sure if you liked "%s". Tell me more about "%s".' % (movie, movie)
          return response
      return response + '\nTell me about another movie you have seen.'


    #############################################################################
    # 3. Movie Recommendation helper functions                                  #
    #############################################################################

    def scoreSentiment(self, input):
      input = re.sub('"([^"]*)"', '', input)
      score = 0
      total = 0
      negate = 1
      for word in input.split():
        if word == 'not' or word.endswith('n\'t'): negate *= -1
        if word in self.sentiment:
          total += 1
          if self.sentiment[word] == 'pos': score += 1 * negate
          else: score -= 1 * negate
      if total == 0: return 0
      return float(score) / total


    def read_data(self):
      """Reads the ratings matrix from file"""
      # This matrix has the following shape: num_movies x num_users
      # The values stored in each row i and column j is the rating for
      # movie i by user j
      self.titles, self.ratings = ratings()
      reader = csv.reader(open('data/sentiment.txt', 'rb'))
      self.sentiment = dict(reader)
      self.titleIndex = {self.titles[i][0]: i for i in range(len(self.titles))}



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

"""
Author: Kiran Teja Sarvamthota
Title: Alternating Least Square Implementation in Apache Spark for Job Recommendation
Instructions:
	Step 1: Open Hortonworks Spark Virtual box/ spark environment with atleast Spark 1.2.1 and Python 2.7.7 installed.
	Step 2: If required install numpy, scipy libraries.
	Step 3: Execute the command "export SPARK_HOME=/usr/hdp/2.2.4.2-2/spark" in unix server.
	Step 4: Transfer the code file SNAG_ALS.py, snagratings.csv, snagjobs.csv to unix server.
	Step 5: Create a directory in HDFS like /user/snagdata and transfer the .csv files to input HDFS directory created.
	Step 6: To run the recommendation program execute: spark-submit SNAG_ALS.py 
	Step 7: An output files is created in the same directory with userID,recommendedProduct,predictedRating
"""


# Implementation of Job recommendation system on spark
from __future__ import print_function

import sys
import itertools
import numpy as np
from numpy.random import rand
from numpy import matrix
from pyspark import SparkConf, SparkContext
from math import sqrt
from operator import add
from os.path import join, isfile, dirname
import time
start = time.time()



def parseRating(line):
    """
    Parses a rating record format userId,jobId,rating,s.no .
    """
    fields = line.strip().split(",")
    return long(fields[3]) % 10, (str(fields[0]), str(fields[1]), float(fields[2]))

def parseJob(line):
    """
    Parses a job record in format jobId,jobTitle .
    """
    fields = line.strip().split(",")
    return fields[0], fields[1]


def rmse(R, usermatrix, jobmatrix):
    """
    Compute RMSE (Root Mean Squared Error).
    """
    Rp = np.dot(usermatrix,jobmatrix.T)
    err = R - Rp
    errsq = np.power(err, 2)
    mean = (np.sum(errsq))/(M * U)
    return np.sqrt(mean)
    
   
# function for computing the values and updating user matrix
def updateUser(i, jobmat, ratings):
    uu = jobmat.shape[0]
    ff = jobmat.shape[1]

    XtX = jobmat.T * jobmat
    Xty = jobmat.T * ratings[i, :].T

    for j in range(ff):
        XtX[j, j] += LAMBDA * uu
    #solves a linear matrix equation
    return np.linalg.solve(XtX, Xty)

# function for computing the values and updating job matrix

def updateJob(i, usermat, ratings):
    uu = usermat.shape[0]
    ff = usermat.shape[1]

    XtX = usermat.T * usermat
    Xty = usermat.T * ratings[i, :].T

    for j in range(ff):
        XtX[j, j] += LAMBDA * uu

    return np.linalg.solve(XtX, Xty)

	
LAMBDA = 0.01   # regularization
# makes the random numbers predictable
np.random.seed(42)	
	
if __name__ == "__main__":

    print("Running Job Recommendation system using ALS")
    conf = SparkConf()\
    .setAppName("MovieLensWithALS")\
    .set("spark.executor.memory", "2g")
    sc = SparkContext(conf=conf)
    
    # load ratings and job titles

   #created a directory in HDFS
    snagHomeDir = "/user/snagdata"


    # ratings is an RDD of (last digit of timestamp, (userId, movieId, rating))
    ratings = sc.textFile(join(snagHomeDir, "snagratings.csv")).map(parseRating)

    jobs = dict(sc.textFile(join(snagHomeDir, "snagjobs.csv")).map(parseJob).collect())

    job_ids = jobs.keys()

    job_ids = np.sort(job_ids)

    numRatings = ratings.count()
    #print(numRatings)
    numUsers = ratings.values().map(lambda r: r[0]).distinct().count()
    numJobs = ratings.values().map(lambda r: r[1]).distinct().count()

    print ("Got %d ratings from %d users on %d jobs." % (numRatings, numUsers, numJobs))

    numPartitions = 5

    
    training = ratings.values().repartition(numPartitions).cache()

    
    a_list =training.collect()


    a_array = np.array(a_list)
    # Returns the sorted unique elements and indices of an array.

    rows, row_pos = np.unique(a_array[:, 0], return_inverse=True)
    cols, col_pos = np.unique(a_array[:, 1], return_inverse=True)

    user_array=list(np.unique(a_array[:, 0]))



    pivot_table = np.zeros((len(rows), len(cols)), dtype=float)
    pivot_table[row_pos, col_pos] = a_array[:, 2]

    t_matrix = np.matrix(pivot_table)

    print(t_matrix)
		
    U =  numUsers # number of users
    M =  numJobs # number of jobs
    F =  10
    ITERATIONS = 20
    partitions = 5

    R = t_matrix # Rating matrix
    W = R>0.5 # Initializing the weighted Matrix
    # print(W)
    W[W == True]= 1
    W[W == False]= 0
    print(W)

    # Initializing the Factors
    usermatrix = matrix(rand(U, F)) 
    jobmatrix = matrix(rand(M, F))
    # Broadcasting the Matrices
    Rb = sc.broadcast(R)
    userb = sc.broadcast(usermatrix)
    jobb = sc.broadcast(jobmatrix)

    for i in range(ITERATIONS):
        # parallelizing the computation
        usermatrix = sc.parallelize(range(U), partitions) \
               .map(lambda x: updateUser(x, jobb.value, Rb.value)) \
               .collect()

        
        # arranging it into a matrix 
        usermatrix = matrix(np.array(usermatrix)[:, :, 0])

        # Broadcasting the matrix
        userb = sc.broadcast(usermatrix)
        # parallelizing the computation
        jobmatrix = sc.parallelize(range(M), partitions) \
               .map(lambda x: updateJob(x, userb.value, Rb.value.T)) \
               .collect()
        # arranging into a matrix form
        jobmatrix = matrix(np.array(jobmatrix)[:, :, 0])

        # Broadcasting the matrix
        jobb = sc.broadcast(jobmatrix)
        # getting the error rate
        error = rmse(R, usermatrix, jobmatrix)
        print("Iteration %d:" % i)
        print("\nRMSE: %5.4f\n" % error)
    
    
    recommendation=np.dot(usermatrix,jobmatrix.T)
  

    output=open("snagfulloutput.txt",'w')

    #Giving Recommendations 
    for i in range(U):
        #Returns the indices that would sort an array.
        
        indices = np.array(np.argsort(recommendation[i,:]))
        #Return a copy of the array collapsed into one dimension
        indices = indices.flatten()
        number_of_recs = 5
        string_recommend="-------------------Recommendations for user"+"  "+str(user_array[i])+"--------------------------"+"\n"
        for index in indices[::-1]:
            if ~W[i, index]:
                string_recommend=string_recommend+"Job title"+" :  "+jobs[job_ids[index]]+","+"Job ID"+" :  "+str(job_ids[index])+"\n"
                number_of_recs -= 1
            if number_of_recs == 0:
                break
    	final_string=string_recommend.encode('utf-8', 'ignore')
        output.write(final_string)
        
    end = time.time()
    print("total running time = %.2f minutes" % ((end - start)/60))
    sc.stop()

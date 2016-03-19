#!/usr/bin/spark-submit
#
# Problem Set #4
# Implement wordcount on the shakespeare plays as a spark program that:
# a.Removes characters that are not letters, numbers or spaces from each input line.
# b.Converts the text to lowercase.
# c.Splits the text into words.
# d.Reports the 40 most common words, with the most common first.

# Note:
# You'll have better luck debugging this with ipyspark
import matplotlib.pyplot as plt
import sys
from operator import add
from pyspark import SparkContext
import datetime

if __name__ == "__main__":
    
    ##
    ## Parse the arguments
    ##
    sc = SparkContext( appName="Wikipedia Count" )

    infile1 =  's3://gu-anly502/maxmind/GeoLite2-Country-Blocks-IPv4.csv'
    infile2 =  's3://gu-anly502/maxmind/GeoLite2-Country-Locations-en.csv'
    line1 = sc.textFile(infile1)
    line2 = sc.textFile(infile2)

    ip = line1.map(lambda ipline:(ipline.split(",")[1],ipline.split(",")[0]))
    country = line2.map(lambda ctline:(ctline.split(",")[0],ctline.split(",")[5]))

    ## 
    ## Run WordCount on Spark
    ##
    ip_by_country=ip.join(country).collect()
    ipbyct=[]
    for (key,value) in ip_by_country:
	ipbyct.append(value)
    ipbyct2=[]
    for (ip,country) in ipbyct:
        ip2=ip.split('/')[0]
        ip3=ip2.split('.')
	ip4=''.join(ip3)
	ipbyct2.append((ip4,country))
    with open("ip_by_country2.txt","w") as fout:
        for (ip,country) in ipbyct2:
            fout.write("{}\t{}\n".format(ip,country))
    sc.stop()

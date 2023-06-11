# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 14:27:20 2021

@author: reece
"""

import math
import pylab
import random


#RUN PROGRAM FIRST BEFORE YOU TWEAK

#TWEAK THESE PARAMETERS ONLY:--------------------------------------------------
    
#for the graphs:
xmin = -3
xmax = 10
ymin = -3
ymax = 10

#number of clusters and number of iterations per type
number_of_clusters = 3
num_per_trial = 20

#for the mean and spread of the data points(uses random.gauss to choose points)
mean_s1x, mean_s1y = 2, 3
std_s1x, std_s1y = 1, 1

mean_s2x, mean_s2y = 5, 5
std_s2x, std_s2y = 0.7, 1

mean_s3x, mean_s3y = 7, 7
std_s3x, std_s3y = 1, 1

#if you dont want to see the loading graphics, change this to False:
print_while_loading = True

#if you want to get the same graph on repeat, uncomment the next line and feel free to change the 50 to another number
#random.seed(50)

#DONT TWEAK BELOW THIS---------------------------------------------------------

for i in range(5):
    print('')
print('welcome to k-means clustering demonstration, designed by Reece Shuttleworth')


class Point():
    def __init__(self, xval, yval):
        self.xval=xval
        self.yval=yval
        self.location=(xval, yval)
    
    def GetX(self):
        return self.xval
    
    def GetY(self):
        return self.yval
    
    def GetDistance(self, other):
        otherx, othery = other.GetX(), other.GetY()
        return math.sqrt((self.xval-otherx)**2+(self.yval-othery)**2)
    
    
class Cluster():
    def __init__(self, initial_point):
        self.centroid=Point(initial_point.GetX(), initial_point.GetY())
        self.points=[initial_point]
        
    def GetCentroid(self):
        return self.centroid
        
    def AddPoint(self, new_point):
        self.points.append(new_point)
      
    def ResetPoints(self):
        self.points=[]
        
    def GetPoints(self):
        return self.points
    
    def UpdateCentroid(self):
        """  
        if unchanged: return False, else Return True
        """
        xymean=pylab.array([0.,0.])
        for point in self.points:
            point = pylab.array([point.GetX(), point.GetY()])
            xymean += point
        xymean /= len(self.points)
        if xymean[0] != self.centroid.GetX() or xymean[1] != self.centroid.GetY():
            self.centroid = Point(xymean[0], xymean[1])
            return True
        return False
    

points=[]
xvals1=[]
yvals1=[]
xvals2=[]
yvals2=[]
for point in range(num_per_trial):
    x=random.gauss(mean_s1x, std_s1x)
    y=random.gauss(mean_s1y, std_s1y)
    points.append(Point(x, y))
    xvals1.append(x)
    yvals1.append(y)
    
for point in range(num_per_trial):
    x=random.gauss(mean_s2x, std_s2x)
    y=random.gauss(mean_s2y, std_s2y)
    points.append(Point(x, y))
    xvals2.append(x)
    yvals2.append(y)
    
pylab.title('actual clusters')
pylab.xlim(xmin, xmax)
pylab.ylim(ymin, ymax)
pylab.plot(xvals1, yvals1, 'r.', markersize=10)
pylab.plot(xvals2, yvals2, 'b.', markersize=10)

xvals3=[]
yvals3=[]
for point in range(num_per_trial):
    x=random.gauss(mean_s3x, std_s3x)
    y=random.gauss(mean_s3y, std_s3y)
    points.append(Point(x, y))
    xvals3.append(x)
    yvals3.append(y)
pylab.plot(xvals3, yvals3, 'y.', markersize=10)
pylab.show()
    



def map_them(points, title = 'what it looks like without clusters'):
    xpoints=[point.GetX() for point in points]
    ypoints=[point.GetY() for point in points]
    pylab.title(title)
    pylab.plot(xpoints, ypoints, 'r^')
    pylab.xlim(xmin, xmax)
    pylab.ylim(ymin, ymax)
    pylab.show()

def graph_clusters(clusters, title = 'loading'):
    graph_colors=['k.', 'g.', 'b.', 'y.', 'r.', 'k^', 'g^', 'b^', 'y^', 'r^']
    pylab.title(title)
    for cluster in clusters:
        color=graph_colors.pop()
        xvals=[cluster.GetPoints()[i].GetX() for i in range(len(cluster.GetPoints()))]
        yvals=[cluster.GetPoints()[i].GetY() for i in range(len(cluster.GetPoints()))]
        pylab.plot(xvals, yvals, color)
        pylab.xlim(xmin, xmax)
        pylab.ylim(ymin, ymax)
    pylab.show()
        
map_them(points)


def find_clusters(points, numclusters, print_loading = True):
    points_=points[:]
    random.shuffle(points_)
    clusters=[Cluster(points_.pop()) for i in range(numclusters)]
    graph_clusters(clusters, 'initial points randomly selected as centroids')
    while True:
        for point in points:
            mindist=point.GetDistance(clusters[0].GetCentroid())
            index=0
            for i in range(1, len(clusters)):
                if point.GetDistance(clusters[i].GetCentroid())<mindist:
                    mindist=point.GetDistance(clusters[i].GetCentroid())
                    index=i
            clusters[index].AddPoint(point)
        test_for_change=[cluster.UpdateCentroid() for cluster in clusters]
        if True not in test_for_change:
            #enters if no change is made to any cluster
            break
        if print_loading:
            graph_clusters(clusters)
        for cluster in clusters:
            cluster.ResetPoints()
    return clusters


clusters=find_clusters(points, number_of_clusters, print_while_loading) 
graph_clusters(clusters, 'determined clusters')
        
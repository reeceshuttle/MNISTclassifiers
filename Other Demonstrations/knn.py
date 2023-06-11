# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 15:19:56 2021

@author: reece
"""

import math
import random
import pylab
import statistics

#FEEL FREE TO CHANGE ANYTHING BELOW THIS LINE----------------------------------

#if you want to change the mean or standard deviation of the two distributions
Ax_mean = 1
Ax_std = 2.5
Ay_mean = 1
Ay_std = 2.5

Bx_mean = 4
Bx_std = 2
By_mean = 4
By_std = 2

num_trials = 2500
number_neighbors = 60
kmin = 1
kmax = 21

random.seed(100)

#DONT CHANGE ANYTHING BELOW HERE-----------------------------------------------

print('\n\n\n\n\nk-nearest neighbors demonstration by Reece Shuttleworth')

class Point():
    def __init__(self, x, y, label):
        self.x=x
        self.y=y
        self.label=label
    
    def GetX(self):
        return self.x
    
    def GetY(self):
        return self.y
    
    def GetLabel(self):
        return self.label
    
    def GetDistance(self, other):
        return math.sqrt((self.x-other.GetX())**2+(self.y-other.GetY())**2)
    
def generate_neighbors(number):
    points=[]
    Ax=[random.gauss(Ax_mean, Ax_std) for x in range(number)]
    Ay=[random.gauss(Ay_mean, Ay_std) for y in range(number)]
    for num in range(len(Ax)):
        points.append(Point(Ax[num], Ay[num], 'A'))
    
    Bx=[random.gauss(Bx_mean, Bx_std) for x in range(number)]
    By=[random.gauss(By_mean, By_std) for y in range(number)]
    for num in range(len(Bx)):
        points.append(Point(Bx[num], By[num], 'B'))
    return points, Ax, Ay, Bx, By

def knearest_neighbors(point_to_label, neighbors, k):
    """point_to_label is a Point object
        neightbors is a list of Point objects
        k is int
        returns estimated label based on k nearest neighbors"""
    if k>len(neighbors):
        raise Exception('dude what the fuck, too many k values')
    neighborhood=neighbors[:]
    knearest=[]
    distances=[]
    for i in range(k):
        point=neighborhood.pop()
        knearest.append(point)
        distances.append(point_to_label.GetDistance(point))
    for i in range(len(neighborhood)):
        if point_to_label.GetDistance(neighborhood[i]) < max(distances):
            index=distances.index(max(distances))
            knearest[index]=neighborhood[i]
            distances[index]=point_to_label.GetDistance(neighborhood[i])
    label_guesses=[]
    for point in knearest:
        label_guesses.append(point.GetLabel())
    return statistics.mode(label_guesses)

def graph(x, y, title, annotate=False, label=''):
    """x and y are lists of lists"""
    pylab.title(title)
    color=['r^', 'b^', 'y^']
    for i in range(len(x)):
        pylab.plot(x[i], y[i], color[i])
    if annotate:
        pylab.annotate(label, (x[-1], y[-1]))
    pylab.axhline(0)
    pylab.axvline(0)
    pylab.show()

def evaluate_best_k(means, stds, num_trials, num_neighbors, kmin=1, kmax=11, verbosity=False):
    """means is a list of means in order
        stds is a list of stds in order"""
    neighbors, Ax, Ay, Bx, By = generate_neighbors(num_neighbors)
    kval=[]
    kval_accuracy=[]
    graph([Ax, Bx], [Ay, By], 'setup')
    print('this may take a little while to run(just a headsup)')
    for k in range(kmin, kmax+1, 2):
        success=0
        total=0
        kval.append(k)
        for trial in range(num_trials):
            if random.random()<0.5:
                tolabel_x = random.gauss(means[0], stds[0])
                tolabel_y = random.gauss(means[1], stds[1])
                tolabel_label = 'A'
            else:
                tolabel_x = random.gauss(means[2], stds[2])
                tolabel_y = random.gauss(means[3], stds[3])
                tolabel_label = 'B'
            if knearest_neighbors(Point(tolabel_x, tolabel_y, tolabel_label), neighbors, k) == tolabel_label:
                success+=1
            total+=1
            if verbosity:
                graph([Ax, Bx, tolabel_x], [Ay, By, tolabel_y], 'trial {}; random point in yellow, letter indicates which group its from'.format(trial+1), True, tolabel_label)
        kval_accuracy.append(success/total)
    return kval, kval_accuracy

x,y=evaluate_best_k([Ax_mean, Ay_mean, Bx_mean, By_mean], [Ax_std, Ay_std, Bx_std, By_std], num_trials, number_neighbors, kmin, kmax)

pylab.title('k vs accuracy')
pylab.plot(x, y, 'r-')
pylab.xlabel('k value')
pylab.ylabel('accuracy')
pylab.show()

for i in range(5):
    print('')
print('new version, this time to see each random point selected:')
x,y=evaluate_best_k([Ax_mean, Ay_mean, Bx_mean, By_mean], [Ax_std, Ay_std, Bx_std, By_std], 15, 20, 1, 1, True)
    
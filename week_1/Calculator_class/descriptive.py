#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 14:25:52 2019

@author: swilson5
"""
import math



class Calculator:
    def __init__(self, data):
        self.dataset = data

    @property
    def length(self):
        return len(self.dataset)

    @property
    def mean(self):
        total = 0
        for data in self.dataset:
            total += data
        return total / len(self.dataset)

    @property
    def median(self):
        ordered = sorted(self.dataset)
        length = self.length
        if length % 2:
            n = int((length - 1) / 2)
            return ordered[n]
        else:
            m1 = int(length / 2)
            m2 = m1 - 1
            return (ordered[m1] + ordered[m2]) / 2

    @property
    def variance(self):
        total = 0
        for datum in self.dataset:
            total += (datum - self.mean) ** 2
        return total / (len(self.dataset) - 1)

    @property
    def stand_dev(self):
        return math.sqrt(self.variance)

    @property
    def quartiles(self):
        ordered = sorted(self.dataset)
        length = self.length
        quart_odd = lambda x, y: [ordered[x], ordered[y]]
        quart_even = lambda x1, x2, y1, y2: [(ordered[x1]+ordered[x2])/2, (ordered[y1]+ordered[y2])/2]
        if length % 2 and (length - 1) % 4:
            q1n = int((length - 3 ) / 4)
            q3n = length-(q1n) - 1
            return quart_odd(q1n, q3n)
        if length % 2 and not (length - 1) % 4:
            q1na = int((length - 1 ) / 4)
            q1nb = q1na - 1
            q3na = length - q1na - 1
            q3nb = q3na + 1
            return quart_even(q1na, q1nb, q3na, q3nb)
        if not length % 2 and (length) % 4:
            q1n = int((length - 2) / 4)
            q3n = length-(q1n) - 1
            return quart_odd(q1n, q3n)
        if not length % 4:
            q1na = int((length) / 4)
            q1nb = q1na - 1
            q3na = length - (q1na) - 1
            q3nb = q3na + 1
            return quart_even(q1na, q1nb, q3na, q3nb)

    @property
    def iqr(self):
        return self.quartiles[1] - self.quartiles[0]

    def add_data(self, data):
        self.dataset.extend(data)

    def remove_data(self, deletions):
        new_list = []
        for data in self.dataset:
            if data not in deletions:
                new_list.append(data)
        self.dataset = new_list

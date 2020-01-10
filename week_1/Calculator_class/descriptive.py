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

    def add_data(self, data):
        self.dataset.extend(data)

    def remove_data(self, deletions):
        new_list = []
        for data in self.dataset:
            if data not in deletions:
                new_list.append(data)
        self.dataset = new_list

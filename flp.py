#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""Module for FLP utilities.
Author: Seth Phillips
Date: 2 October 2020
Forked from Project 5
"""

from collections import namedtuple
import math
import time
import random
from typing import Tuple
from matplotlib import pyplot as plt

Point = namedtuple("Point", "x y")
HIGHLIGHT_COLOR = "#e11f26"

class FlpHandler:
	
	file: str = ""
	data: dict = {}
	
	# Initialize the class by setting and reading the FLP file
	def __init__(self, file: str) -> None:
		self.set_file(file)
		self.data = self.read_flp_file()
	
	# Get data from this instance
	def get_data(self) -> dict:
		return self.data
	
	# Get the file from this instance
	def get_file(self) -> str:
		return self.file
	
	# Set the file instance
	def set_file(self, file: str) -> None:
		self.file = file
	
	# Parse a FLP-formatted file (comparable to the previous experiment)
	def read_flp_file(self) -> dict:
		"""Parse a FLP-formatted file and return a dictionary of available fields."""
		
		if self.file == "":
			raise NameError("file parameter not passed")
		
		result: dict = {"NODE_COORD_SECTION": []}
		x_min: float = None
		x_max: float = None
		y_min: float = None
		y_max: float = None
		
		try:
			
			# Loop through line by line
			for line in open(self.file, 'r'):
				
				# Separate data
				line = line.rstrip('\n')
				keyvalues: list = line.split(':')
				coords: list = line.split(' ')
				line.split(' ')
				
				if len(keyvalues) > 1:
					# Add a key/value pair directly
					result[keyvalues[0]] = ((result[keyvalues[0]] + '\n' + keyvalues[1]) if keyvalues[0] in result else keyvalues[1]).lstrip()
					continue
				elif len(coords) == 3:
					# Or add a coordinate to the coord section
					x = float(coords[1])
					y = float(coords[2])
					result["NODE_COORD_SECTION"].append(Point(x, y))
					
					# Verify absolute minimums and maximums
					if x_max is None or x > x_max:
						x_max = x
					if x_min is None or x < x_min:
						x_min = x
					if y_max is None or y > y_max:
						y_max = y
					if y_min is None or y < y_min:
						y_min = y
			
			# Strong-type any applicable data fields
			result["DIMENSION"] = int(result["DIMENSION"])
			
			# Handle extra instructions
			result["EXTRA"] = result["EXTRA"].split() if "EXTRA" in result else []
			if "FLIP_AXES" in result["EXTRA"]:
				result["NODE_COORD_SECTION"] = [Point(x=p.y, y=p.x) for p in result["NODE_COORD_SECTION"]]
				x_max, y_max = y_max, x_max
				x_min, y_min = y_min, x_min
			
			# Add in absolutes
			result["COORD_MIN"] = Point(x_min, y_min)
			result["COORD_MAX"] = Point(x_max, y_max)
			
			# Determine an appropriate variance
			result["X_RESOLUTION"] = (x_max - x_min) / result["DIMENSION"]
			result["Y_RESOLUTION"] = (y_max - y_min) / result["DIMENSION"]
			
		except OSError as e:
			raise OSError("could not open FLP file for reading: " + e)
		
		except IndexError as e:
			raise IndexError("malformed FLP file: " + e)
		
		if len(result["NODE_COORD_SECTION"]) != result["DIMENSION"]:
			raise OSError("dimension does not match supplied coordinates")
		
		# Return all sections
		return result


class Timings:
	
	timer: list = []
	precision: int = 0
	
	# Initialize the timer
	def __init__(self, precision = 5) -> None:
		self.set_precision(precision)
		self.clear_timer()
	
	# Clear the timer
	def clear_timer(self) -> None:
		self.timer: list = []
	
	# Set the print precision
	def set_precision(self, precision) -> None:
		self.precision: int = precision
	
	# Append a new timer value
	def add(self, name: str, mark: str = "") -> None:
		self.timer.append((name, time.clock(), mark))
	
	# Pretty print all timer values, as well as symmetric differences and marked pairs
	def __str__(self) -> None:
		
		# Print the raw values
		result: str = "\nTime Values:\n"
		has_mark: bool = False
		for time in self.timer:
			result += "\t" + time[0] + ": " + str(round(time[1], self.precision)) + "\n"
			if time[2] != "":
				has_mark = True
		
		# Print the symmetric differences (i.e. 123321 -> 33, 22, 11)
		result += "\nAnalysis:\n"
		offset: int = 0
		midpoint: int = int(len(self.timer)/2)
		for i in range(midpoint, len(self.timer)):
			if midpoint-1-offset < 0:
				break
			lower = self.timer[midpoint-1-offset]
			upper = self.timer[i]
			result += "\t\u0394(" + lower[0] + "," + upper[0] + "): " + str(round(upper[1] - lower[1], self.precision)) + "s\n"
			offset += 1
		
		# Print any special differences from their previous value
		if has_mark:
			result += "\nMarked Values:\n"
			for i in range(1, len(self.timer)):
				if self.timer[i][2] != "":
					result += "\t" + self.timer[i][2] + ": " + '{:f}'.format(round(self.timer[i][1] - self.timer[i-1][1], self.precision)) + "s\n"
		
		return result

class Fitness:
	"""Class to handle a map's total distance and fitness"""

	def __init__(self, mapping):
		"""Initialize all variables"""
		
		self.mapping = mapping
		self.distance = 0
		self.fitness = 0.0		

	def map_distance(self):
		"""Calculate the exact, total map distance"""
		
		# If not already determined, iterate through all facilities for the total distance
		if self.distance == 0:
			self.distance = 0
			for i in self.mapping:
				for j in range(len(self.mapping[i])):
					self.distance += self.mapping[i][j][1]
			
		return self.distance

	def map_fitness(self):
		"""Calculate the fitness of the map"""
		
		# If not already determined, calculate fitness where minimum is better (i.e. inverse)
		if self.fitness == 0:
			self.fitness = 1 / float(self.map_distance())
		return self.fitness


def assign_clients(clients: list, facilities: list):
	"""Map each client to the facility it's closest to via index"""

	result = {x: [] for x in facilities}
	
	for i in range(len(clients)):
		best_facility = None
		best_value = None
		for j in range(len(facilities)):
			approximation = dirty_distance(clients[i], facilities[j])
			if best_facility is None or approximation < best_value:
				best_facility = facilities[j]
				best_value = approximation
		result[best_facility].append((clients[i], math.sqrt(best_value)))
	
	return result

def dirty_distance(p1: Point, p2: Point) -> float:
	"""Get a magnitude of distance squared (for quick ranking)"""
	
	dx = p1.x - p2.x
	dy = p1.y - p2.y
	
	return dx*dx + dy*dy

def plot_points(FLP, facilities = None):
	"""
	Plot the points from given FLP data.
	
	FLP -- A FLP object
	facilities -- An optional list of facilities to additionally render
	"""
	data = FLP.get_data()
	fig = plt.figure()
	fig.canvas.set_window_title("%s Point View" % data["NAME"])

	if facilities is None:
	
		# Set up data points
		x_data = [x for x, y in data["NODE_COORD_SECTION"]]
		y_data = [y for x, y in data["NODE_COORD_SECTION"]]
	
		# Only show client points
		plt.plot(x_data, y_data, "o", color=HIGHLIGHT_COLOR)
		for i in range(len(data["NODE_COORD_SECTION"])):
			plt.text(data["NODE_COORD_SECTION"][i].x, data["NODE_COORD_SECTION"][i].y, str(i+1))
		plt.suptitle(data["COMMENT"])
		
	else:
	
		# Show data points and facilities
		color = ["#" + "".join([random.choice("0123456789ABCD") for j in range(6)]) for i in range(len(facilities))] if len(facilities) > 1 else ["#000000"]
		clusters = assign_clients(data["NODE_COORD_SECTION"], facilities)
		plt.rcParams.update({"font.size": 7, "figure.titlesize": 12, "axes.labelsize": 11})
		
		i = 0
		totals = len(facilities) * [0]
		for facility in clusters.keys():
			for client in clusters[facility]:
				plt.plot((client[0].x, facility.x), (client[0].y, facility.y), "o-", color=color[i], linewidth=1)
				totals[i] += client[1]

			plt.plot(facility.x, facility.y, "o", color=color[i], label=r"F%d $\Sigma x$=%.2f $\bar x$=%.2f" % (i+1, totals[i], totals[i]/(1 if len(clusters[facility])==0 else len(clusters[facility]))))
			plt.text(facility.x, facility.y, "(%.2f, %.2f)" % (facility.x, facility.y))
			i += 1
		
		plt.legend(loc="upper left")
		plt.plot([x for x, y in facilities], [y for x, y in facilities], "o", color=HIGHLIGHT_COLOR)
		plt.suptitle("Total Cost: %.3f\nAverage Cost: %.3f" % (sum(totals), sum(totals)/data["DIMENSION"]))
	
	plt.show()

def plot_improvement(generation_results, percent_reduced, name):
	"""Plot the improvement chart for a given set of generations."""

	# Make a new graphical window
	fig = plt.figure()
	fig.canvas.set_window_title("FLP Improvement Analysis: " + name)
	final_dist_index = len(generation_results) - 1

	# Plot each generation's performance
	plt.plot([x[1] for x in generation_results], color=HIGHLIGHT_COLOR)

	# Show the final result and label the graph
	plt.text(final_dist_index*0.8, generation_results[final_dist_index][1], "~%d" % int(generation_results[final_dist_index][1]), fontsize=9)
	plt.ylabel("Distance")
	plt.xlabel("Generation")
	plt.suptitle("Reduction: %.3f%%" % percent_reduced)

	# Finally, show the plot
	plt.show()
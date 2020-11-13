#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""Script to run a Genetic Algorithm with Wisdom of a Crowd on a Facility Location Problem.
Author: Seth Phillips
Date: 21 October 2020
Forked from Project 4
"""

##########################
# PROGRAM INITIALIZATION #
##########################

def missing_package(package: str) -> None:
	print("[*] Your system is missing %(0)s. Please run `easy_install %(0)s` or `pip install %(0)s` before executing." % { '0': package })
	exit()

# Begin imports
import argparse
import copy
from collections import Counter
from decimal import Decimal as D
import os
import operator
import platform
import random
import sys

try:
	from packaging import version
except ImportError:
	missing_package("packaging")

try:
	from matplotlib import pyplot as plt
	from matplotlib import animation
except ImportError:
	missing_package("matplotlib")

try:
	import pandas as pd
except ImportError:
	missing_package("pandas")

try:
	import numpy as np
except ImportError:
	missing_package("numpy")

try:
	import progressbar
except ImportError:
	missing_package("progressbar")

try:
	import flp
except ImportError as e:
	print("[*] Your system is missing the included FLP library file. Please find the original source and download before executing.")
	exit()

if version.parse(platform.python_version()) < version.parse("3.6"):
	print("[*] You must run this on Python 3.6!")
	exit()

# Parse applicable command line arguments
cli = argparse.ArgumentParser(description = "Run a genetic algorithm with wisdom of the crowds to solve a Facility Location Problem.")
cli.add_argument("file", help = "String :: the coordinates input file in FLP-format", type = str, default = "")
cli.add_argument("-n", "--facilities", help = "int :: The exact number of facilities to place", type = int, required = True)
cli.add_argument("-p", "--population-size", help = "int :: The size of each population pool", type = int, required = True)
cli.add_argument("-e", "--elites", help = "int :: The number of elites picked from each population pool", type = int, required = True)
cli.add_argument("-m", "--mutation-rate", help = "int :: The percent of time a mutation will happen (between 0 and 100)", type = int, required = True)
cli.add_argument("-g", "--generations", help = "int :: The number of generations to loop through", type = int, required = True)
cli.add_argument("-w", "--wisdom", help = "bool :: Apply a Wisdom of the Crowds effect to elites on each generation", action = "store_true", default = False)
cli.add_argument("-s", "--skip-improvement", help = "bool :: Don't show the improvement curve at the end of the program", action = "store_true", default = False)
cli.add_argument("-v", "--verbose", help = "bool :: Print every generation's distance (typically for debugging)", action = "store_true", default = False)
ARGS: dict = cli.parse_args()


##########################
#   PROGRAM  FUNCTIONS   #
##########################

def input_error(message: str) -> None:
	"""Send an error message to the user with usage help."""
	
	global cli
	cli.print_usage(sys.stderr)
	print("\n" + os.path.basename(__file__) + ": error: " + message, file=sys.stderr)
	sys.exit(1)

def spawn_population(FLP):
	"""Make an initial population with a given FLP object."""

	data = FLP.get_data()
	results = []

	# Decide a random set of genes for each original chromosome
	for i in range(ARGS.population_size):
		facilities = []
		for j in range(ARGS.facilities):
			facilities.append(flp.Point(x=random.uniform(data["COORD_MIN"].x, data["COORD_MAX"].x), y=random.uniform(data["COORD_MIN"].y, data["COORD_MAX"].y)))
		results.append(facilities)

	return results

def rank_routes(population, FLP):
	"""Rank facilities by their fitness (total distance)."""
	
	clients = FLP.get_data()["NODE_COORD_SECTION"]
	results = {}
	
	# Calculate and store the fitness of all chromosomes
	for i in range(len(population)):
		results[i] = flp.Fitness(flp.assign_clients(clients, population[i])).map_fitness()
	
	# Return the list of chromosomes from best to worst in terms of fitness
	return sorted(results.items(), key = operator.itemgetter(1), reverse = True)

def selection(ranked_population):
	"""Select the next generation's parent sample."""
	
	results = []
	
	# Make a new Pandas dataframe
	df = pd.DataFrame(np.array(ranked_population), columns=["Index", "Fitness"])
	
	# Determine the cumulative sum and percent
	df["cum_sum"] = df.Fitness.cumsum()
	df["cum_perc"] = 100*df.cum_sum/df.Fitness.sum()

	# Ensure the best chromosomes are selected as parents
	for i in range(ARGS.elites):
		results.append(ranked_population[i][0])
	
	# Remaining parents are picked
	for i in range(len(ranked_population) - ARGS.elites):
		pick = 100*random.random()
		
		# Only allow picks where the percent meets a uniformly distributed random choice
		for j in range(len(ranked_population)):
			
			# This essentially adds a weight to parents to naturally pick more fit ones
			if pick <= df.iat[j, 3]:
				results.append(ranked_population[j][0])
				break

	return results

def mating_pool(population, selection_results):
	"""Make a mating pool from a given population"""
	
	results = []
	
	# Iterate over all selected parents indices and append their chromosome to the pool
	for i in range(len(selection_results)):
		results.append(population[selection_results[i]])
	
	return results

def breed(father, mother, keep_gene):
	"""Crossover between two parents by random shuffling segments"""

	# Pick a random subset of genes to swap between parents
	size = len(father)
	geneA = int(random.random() * size)
	geneB = int(random.random() * size)
	start = min(geneA, geneB)
	end = max(geneA, geneB)

	# Treat the father's gene segment as dominant
	# Then fill in the remaining spots with the mother's genes (in order)
	child = mother[0:start] + father[start:end] + mother[end:size]
	
	# Make sure this child has the wisdom of the crowds gene, if applicable
	if keep_gene is not None:
		
		# Force the gene to exist by overwriting a random spot
		child[int(random.random() * size)] = keep_gene

	return child

def breed_population(matingpool):
	"""Breed the population."""
	global ARGS

	children = []
	pool_size = len(matingpool)
	pool = random.sample(matingpool, pool_size)

	# Directly add the elite parents
	for i in range(ARGS.elites):
		children.append(matingpool[i])
	
	# Find shared nodes in the elites, if applicable
	keep_gene = None
	if ARGS.wisdom and ARGS.elites > 0:
		wise_facilities = []
		for map in children:
			for i in range(len(map)):
				wise_facilities.append(map[i])
		keep_gene = Counter(wise_facilities).most_common(1)[0][0]

	# Breed the remaining casuals
	for i in range(pool_size - ARGS.elites):
		child = breed(pool[i], pool[pool_size - i - 1], keep_gene)
		children.append(child)

	return children

def mutate(population, FLP):
	"""Randomly swap two genes in chromosomes according to the mutation rate"""

	for chromosome in range(len(population)):
	
		# Iterate through each individual chromosome index
		for gene in range(len(population[chromosome])):

			# Determine if the mutation should apply
			if(random.random() < ARGS.mutation_rate / 100):

				data = FLP.get_data()

				# Give variance to the point based on a normal distribution
				new_x = np.random.normal(loc=population[chromosome][gene].x, scale=data["X_RESOLUTION"], size=1)[0]
				new_y = np.random.normal(loc=population[chromosome][gene].y, scale=data["Y_RESOLUTION"], size=1)[0]
				population[chromosome][gene] = flp.Point(new_x, new_y)

	return population

def next_generation(current_gen, FLP):
	"""Determine the next generation from a given current generation"""
	
	# Select parents, breed, mutate, and return
	return mutate(breed_population(mating_pool(current_gen, selection(rank_routes(current_gen, FLP)))), FLP)

def best_chromosome(ranked_population):
	"""Determine the best chromosome index and its distance."""
	
	# Re-invert the fitness of the chromosome with the largest fitness to determine best distance
	return (ranked_population[0][0], 1 / ranked_population[0][1])

def GA(FLP):
	"""Run the genetic algorithm on an FLP object."""
	
	# Spawn the original population
	firstborn = spawn_population(FLP)
	population = firstborn
	results = (ARGS.generations + 1) * [None]
	
	# Record original population data
	chromosome = best_chromosome(rank_routes(firstborn, FLP))
	results[0] = (firstborn[chromosome[0]], chromosome[1])

	# Start the dynamic progress bar
	bar = progressbar.ProgressBar(maxval=ARGS.generations, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
	bar.start()

	# Iterate through each generation, recording the most fit chromosome and breeding the next generation
	for i in range(ARGS.generations):
	
		# Record elites to ensure the best always improves
		keepers = copy.deepcopy([population[x[0]] for x in rank_routes(population, FLP)[:ARGS.elites]])
		
		# Make the next generation
		population = next_generation(population, FLP)
		
		# Inject the elites
		population = population[len(keepers):] + keepers
		
		# Record the results
		chromosome = best_chromosome(rank_routes(population, FLP))
		generation = i+1
		results[generation] = (population[chromosome[0]], chromosome[1])
		
		# Display applicable information to the user
		if ARGS.verbose:
			print("Generation %d: %.2f" % (generation, results[generation][1]))
		bar.update(generation)

	return results

##########################
#    PROGRAM RUNNABLE    #
##########################

def arg_value(arg):
	"""Return a string indicating an arg value"""
	prefix = "is "
	if isinstance(arg, bool):
		return prefix + ("enabled" if arg else "disabled")
	return prefix + str(arg)

def main():
	"""Main program runnable."""

	# Start the program timer
	timer = flp.Timings()
	timer.add("Start")

	# Prepare the data
	print("Number of facilities %s" % arg_value(ARGS.facilities))
	print("Population size %s" % arg_value(ARGS.population_size))
	print("Number of elites %s" % arg_value(ARGS.elites))
	print("Mutation rate %s%%" % arg_value(ARGS.mutation_rate))
	print("Number of generations %s" % arg_value(ARGS.generations))
	print("Wisdom of the Crowds %s" % arg_value(ARGS.wisdom))
	print("\nLoading data from %s..." % ARGS.file, end=" ", flush=True)
	try:
		FLP: FlpHandler = flp.FlpHandler(ARGS.file)
	except Exception as e:
		input_error("can't parse FLP file: " + str(e))
	print("found %i entries." % FLP.data["DIMENSION"])
	timer.add("DataLoad", "Loading data")

	# Run the genetic algorithm
	print("Running genetic algorithm...")
	generations = GA(FLP)
	timer.add("GeneticSim", "Running the simulation")

	# Calculate the disparity to evaluate overall performance
	first_dist = generations[0][1]
	final_map = generations[-1][0]
	final_dist = generations[-1][1]
	reduction = 1 - (final_dist / first_dist)

	# Print all results
	print("\n\nBest of Final Generation %d:" % (len(generations)-1))
	for facility in final_map:
		print("\t(%.3f, %.3f)" % (facility.x, facility.y))
	print("\nFirstborn Total Distance: %.3f\nFirstborn Average Distance: %.3f\n\nFinal Total Distance: %.3f\nFinal Average Distance: %.3f\n\n%.3f%% %s than random start" % (first_dist, first_dist/FLP.data["DIMENSION"], final_dist, final_dist/FLP.data["DIMENSION"], abs(reduction*100), "better" if reduction >= 0 else "worse"))
	timer.add("Finish", "Compiling Results")

	# Show the result as a GUI
	flp.plot_points(FLP, generations[-1][0])
	timer.add("RoutesRender", "Rendering Routes Canvas")

	# Show the improvement curve GUI
	if not ARGS.skip_improvement:
		timer.add("ImproveRenderPre", "User looking at graph")
		flp.plot_improvement(generations, reduction*100, FLP.data["NAME"])
		timer.add("ImproveRenderPost", "Rendering Improvement Canvas")

	# Show timer results
	print(timer)

if __name__ == '__main__':

	# Handle user meta
	try:
		if os.path.isfile(ARGS.file):
			if ARGS.facilities > 0:
				if ARGS.population_size > 1:
					if ARGS.elites > -1:
						if ARGS.elites <= ARGS.population_size:
							if ARGS.mutation_rate in range(0, 101):
								if ARGS.generations > -1:
									main()
								else:
									input_error("number of generations must be a positive integer")
							else:
								input_error("mutation rate must be between 0 and 100 (expressed as a percentage)")
						else:
							input_error("number of elites must be less than or equal to the population size")
					else:
						input_error("number of elites must be a positive integer")
				else:
					input_error("population size must be at least 2")
			else:
				input_error("minimum number of facilities must be 1")
		else:
			input_error("input file not found: \"" + ARGS.file + "\"")

	except KeyboardInterrupt:
		print("\n\nTerminating.")
		sys.exit(0)

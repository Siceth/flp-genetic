#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""Script to generate FLP data in a given range.
Author: Seth Phillips
Date: 8 November 2020
"""

import argparse
import random

cli = argparse.ArgumentParser(description = "Generate Facility Location Problem coordinates.")
cli.add_argument("file", help = "String :: the coordinates output file in FLP-format", type = str, default = "")
cli.add_argument("-n", "--number", help = "int :: The number of points to generate", type = int, required = True)
cli.add_argument("-c", "--clusters", help = "int :: The clustering index (0 = no clustering / maximum variance; n = maximum clustering / minimum variance)", type = int, default = 0)
cli.add_argument("--xmin", help = "float :: The lower generative bound for the x axis", type = float, default = -10.0)
cli.add_argument("--xmax", help = "float :: The upper generative bound for the x axis", type = float, default = 10.0)
cli.add_argument("--ymin", help = "float :: The lower generative bound for the y axis", type = float, default = -10.0)
cli.add_argument("--ymax", help = "float :: The upper generative bound for the y axis", type = float, default = 10.0)
ARGS: dict = cli.parse_args()

def main():
	with open(ARGS.file, "w") as f:
		f.write("NAME: RANDOM%i%s\n" % (ARGS.number, "" if ARGS.clusters==0 else "CLUSTERED"))
		f.write("TYPE: FLP\n")
		f.write("COMMENT: Generated by Point_Generator.py\n")
		f.write("DIMENSION: %i\n" % ARGS.number)
		f.write("EDGE_WEIGHT_TYPE: EUC_2D\n")
		f.write("NODE_COORD_SECTION\n")
		if ARGS.clusters == 0:
			for endpoint in range(1, ARGS.number+1):
				f.write("%i %.6f %.6f\n" % (endpoint, random.uniform(ARGS.xmin, ARGS.xmax), random.uniform(ARGS.ymin, ARGS.ymax)))
		else:
			deviation = ARGS.number / ARGS.clusters
			core = (0.0, 0.0)
			for endpoint in range(1, ARGS.number+1):
				if ARGS.clusters % endpoint == 0:
					core = (random.uniform(ARGS.xmin, ARGS.xmax), random.uniform(ARGS.ymin, ARGS.ymax))
				x_min = core[0] - random.uniform(0, deviation)
				x_max = core[0] + random.uniform(0, deviation)
				y_min = core[1] - random.uniform(0, deviation)
				y_max = core[1] + random.uniform(0, deviation)
				f.write("%i %.6f %.6f\n" % (endpoint, random.uniform(x_min, x_max), random.uniform(y_min, y_max)))


if __name__ == '__main__':
	try:
		main()
	except KeyboardInterrupt:
		print("\n\nTerminating.")
		sys.exit(0)
#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""Script to read FLP data and neatly print to a plot.
Author: Seth Phillips
Date: 8 November 2020
"""

import argparse
import random
import flp

cli = argparse.ArgumentParser(description = "Visually show Facility Location Problem coordinates.")
cli.add_argument("file", help = "String :: the coordinates input file in FLP-format", type = str, default = "")
ARGS: dict = cli.parse_args()

def main():
	FLP = flp.FlpHandler(ARGS.file)
	flp.plot_points(FLP)


if __name__ == '__main__':
	try:
		main()
	except KeyboardInterrupt:
		print("\n\nTerminating.")
		sys.exit(0)
#Ana Margarida Ferro, April 2023
import re
import sys
import argparse
import subprocess as sp
from datetime import datetime
import time


parser = argparse.ArgumentParser(description='Process captions file.')
parser.add_argument('--ti',required=True,help='time stamp initial')
parser.add_argument('--tf', required=True,help='time stamp final')
parser.add_argument('--rt',required=True,help='transcripiton file .rt')
parser.add_argument('--output',required=True,help='transcripiton file .rt')

args = parser.parse_args()


with open(args.rt,'r') as rt:
	f = open(args.output, "w")

	t_initial = datetime.strptime(args.ti.split('.')[0], '%H:%M:%S')
	t_final = datetime.strptime(args.tf.split('.')[0], '%H:%M:%S')


	flag_initial_equal=0
	flag_final_equal=0
	flag_x = 0

	new_t_initial = t_initial
	new_t_final = t_final

	range=0

	for line in rt:
		times = re.findall('\d\d:[0-5]\d:[0-5]\d.\d\d\d',line)
		if len(times) > 0:
			# if this is a times line
			beg_t = times[0]
			beg_t = datetime.strptime(beg_t.split('.')[0], '%H:%M:%S')

			end_t = times[1]
			end_t = datetime.strptime(end_t.split('.')[0],'%H:%M:%S')


			if flag_initial_equal == 0 and beg_t <= t_initial and end_t >= t_initial:
				new_t_initial = times[0]
				f.write(line)
				flag_initial_equal=1
				time = end_t - beg_t
				range += time.seconds

			elif flag_final_equal == 0 and flag_initial_equal ==1 and beg_t <= t_final and end_t >= t_final:
				new_t_final = times[1]
				f.write(line)
				flag_final_equal=1
				time = end_t - beg_t
				range += time.seconds

			elif flag_final_equal == 0 and flag_initial_equal ==1:
					f.write(line)
					time = end_t - beg_t
					range += time.seconds

		else:
			if flag_final_equal == 0 and flag_initial_equal ==1:
				f.write(line)

			elif flag_final_equal == 1 and flag_initial_equal == 1 and flag_x ==0:
				f.write(line)
				flag_x = 1
	f.close()
	print(str(new_t_initial.split('.')[0]) + "," + str(new_t_final.split('.')[0]) + "," + str(range))

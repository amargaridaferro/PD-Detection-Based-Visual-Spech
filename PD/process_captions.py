#April 2023, based on Catarina Botelho work
#Ana Margarida Ferro
import re
import sys
import argparse

parser = argparse.ArgumentParser(description='Process captions file.')
parser.add_argument('--file','-f',required=True,help='Captions file')
parser.add_argument('--type','-t',required=True,choices=['srt','vtt'], help='Captions extension')

args = parser.parse_args()

# auxiliary buffer with frame timestamps
buffer = None

if args.type == 'vtt':

	with open(args.file,'r') as reader:
		for line in reader:

			# read times from line
			times = re.findall('\d\d:[0-5]\d:[0-5]\d.\d\d\d',line)

			if len(times) > 0:
				# if this line has word-level captioning (this is what we want)
				if re.search('<c>',line):
					try:
						# print frame timestamps
						print(times[0] + ' ' + buffer[-1])
						# print transcription only
						print(re.sub('<\d\d:[0-5]\d:[0-5]\d.\d\d\d>|<c>|</c>','',line))
						# clear buffer
						buffer = None
					except:
						raise IOERROR('malformed captions!')
				else:
					buffer = times


elif args.type == 'srt':

	with open(args.file,'r') as reader:
		for line in reader:
			string = line.split()
			if len(string) > 0:
				# ignore caption counter
				if not string[0].isdecimal():
					# if this is timestamp line
					if re.search('\d\d:[0-5]\d:[0-5]\d,\d\d\d',line):
						# re-format time
						times = re.sub(',','.',line)
						print(re.sub('--> |align:start position:0%|\n','',times))
					# this is the transcription
					else:
						print(line)

else:
	raise IOERROR('unknown captions extension.')

from opt_utils import *
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-s", "--skip_compilation", action='store_true', help="skip compilation")
args = parser.parse_args()

if not args.skip_compilation:
	compile_all_opt_examples()

for example in all_examples:
	args = []
	output = run_example(example, args, True).decode('ascii')
	with open(example + ".log", "w") as text_file:
		text_file.write(output)

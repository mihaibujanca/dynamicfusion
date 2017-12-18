import errno
import os
import shutil
import platform
import sys
import os
from utils import *
from glob import glob
from subprocess import call, check_output
s = platform.system()
windows = (s == 'Windows') or (s == 'Microsoft')

script_dir = os.path.dirname(os.path.realpath(__file__)) + "/"
script_output_dir = script_dir + "output/"
opt_src_dir = script_dir + "../API/src/"
examples_dir = script_dir + "../examples/"

all_examples = []
for p in glob(examples_dir + "*/"):
	dir_name = os.path.basename(os.path.normpath(p))
	if dir_name not in ["shared", "external", "data"]:
		all_examples.append(dir_name)
print(all_examples)

def copytree(src, dst):
    if not os.path.exists(dst):
        os.makedirs(dst)
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            copytree(s, d)
        else:
            if not os.path.exists(d) or os.stat(s).st_mtime - os.stat(d).st_mtime > 1:
                shutil.copy2(s, d)

def make_sure_path_exists(path):
	try:
		os.makedirs(path)
	except OSError as exception:
		if exception.errno != errno.EEXIST:
			raise

def ensure_result_dirs():
	make_sure_path_exists(results_dir)


#TODO: Error Handling
# TODO: handle windows/unix differences
def compile_opt_example(example_name):
	os.chdir(examples_dir + example_name)
	if windows:
		VisualStudio(example_name + ".sln", [example_name])
	else:
		call(["make"])
	os.chdir(script_dir)

def compile_all_opt_examples():
	for example in all_examples:
		compile_opt_example(example)

# TODO: handle windows/unix differences
def run_example(example_name, args, return_output=False):
	cwd = os.getcwd()
	os.chdir(examples_dir + example_name)
	print("Running " + example_name)
	cmd = ["x64/Release/" + example_name + ".exe"] + args
	output = ""
	if return_output:
		output = check_output(cmd)
	else:
		call(cmd)
	print("Done with " + example_name)
	os.chdir(cwd)
	return output
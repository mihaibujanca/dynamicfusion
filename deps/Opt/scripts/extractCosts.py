import sys

filename = sys.argv[1]
lines = []
with open(filename, 'r', encoding='utf-16') as text_file:
	lines = text_file.readlines()
indices = [i for i,v in enumerate(lines) if "===" in v]
for i in indices:
	if lines[i+1][:2] == "**":
		print("".join(lines[i:i+4]))
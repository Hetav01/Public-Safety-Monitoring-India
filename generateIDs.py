import csv

infile = '/Users/ajaykumarpatel/Desktop/Data Science/Grad DS Work/DSCI 601 Applied Data Science/old_repos/NLPPunePorsche/PunePorscheTitles_filtered.csv'
outfile = '/Users/ajaykumarpatel/Desktop/Data Science/Grad DS Work/DSCI 601 Applied Data Science/old_repos/NLPPunePorsche/PunePorscheIDs.csv'

with open(infile, "r", newline= '', encoding='utf-16') as infile, open(outfile, "w", newline= '', encoding='utf-16') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)
    
    for row in reader:
        writer.writerow(''.join(row[1].split("=")[1]))

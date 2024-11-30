import csv

infile = '/Users/hetavpatel/Desktop/Data Science/Grad DS Work/DSCI 601 Applied Data Science/old_repos/NLPPunePorsche/PunePorscheTitles.csv'
outfile = '/Users/hetavpatel/Desktop/Data Science/Grad DS Work/DSCI 601 Applied Data Science/old_repos/NLPPunePorsche/PunePorscheIDs.csv'

with open(infile, "r", newline= '', encoding='utf-16') as infile, open(outfile, "w", newline= '', encoding='utf-16') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)  
    
    for row in reader:
        # if (row[1].find("shorts")):
        #     writer.writerow(''.join(row[1].split("s/")[1]))
        # else:
        #     writer.writerow(''.join(row[1].split("=")[1]))
            
        if "/watch?v=" in row[1]:
            writer.writerow(''.join(row[1].split("/watch?v=")[1].split("&")[0]))
        elif "/shorts/" in row[1]:
            writer.writerow(''.join(row[1].split("/shorts/")[1]))
        else:
            None
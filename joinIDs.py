infile = '/Users/hetavpatel/Desktop/Data Science/Grad DS Work/DSCI 601 Applied Data Science/old_repos/NLPPunePorsche/PunePorscheIDs.csv'
outfile = '/Users/hetavpatel/Desktop/Data Science/Grad DS Work/DSCI 601 Applied Data Science/old_repos/NLPPunePorsche/PunePorscheIDsFound.csv'
# Read the input file
with open(infile, 'r', encoding='utf-16') as file:
    lines = file.readlines()

# Write the concatenated rows to a new file
with open(outfile, 'w', encoding='utf-16') as outfile:
    for line in lines:
        # Remove any newline characters and concatenate without spaces
        concatenated_row = ''.join(line.strip().split(','))
        # concatenated_row = concatenated_row.split("&pp")[0]
        outfile.write(concatenated_row + '\n')
        
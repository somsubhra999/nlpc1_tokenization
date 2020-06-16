import nltk
from pathlib import Path
import csv

inpf_path = Path(Path.cwd(), 'data/ticket_Data.csv')
outf_path = Path(Path.cwd(), 'data/ticket_Data_vocab.csv')
has_header = True          # Change this to False if the csv file doesn't have a header

with open(inpf_path, 'r') as inp_csvfile:
    ticket_rdr = csv.reader(inp_csvfile, delimiter=',', quotechar='"')

    with open(outf_path, 'w', newline='\n', encoding='utf-8') as out_csvfile:
        csvwriter = csv.writer(out_csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        if has_header:
            header = next(ticket_rdr)
            csvwriter.writerow(header + ['Vocabulary'])

        for row in ticket_rdr:
            ticket_id, description = row
            tokens = nltk.word_tokenize(description)
            vocab = set(token.lower() for token in tokens)          # Converting to lowercase and removing duplicates
            csvwriter.writerow([ticket_id, description, repr(vocab)])

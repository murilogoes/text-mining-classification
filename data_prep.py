import time
import csv

from utils.CleanText import CleanText


inicio = time.time()

clean_text = CleanText()

with open('data/todos_bos_marco.csv', encoding="utf8") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=';')

    csv_reader.__next__()

    with open('data/todos_bos_tratados_marco.csv', 'w', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(["historico", "contexto"])
        for row in csv_reader:
            row[0] = clean_text.clean(row[0])
            writer.writerow([row[0], str(1 if row[1] == 'Contexto Policial' else 0)])

fim = time.time()

print(fim - inicio)
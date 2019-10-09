import csv

def LOG2CSV(data, csv_file):
    with open(csv_file, "a") as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(data)
    csvFile.close()

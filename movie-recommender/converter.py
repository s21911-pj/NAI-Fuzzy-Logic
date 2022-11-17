import csv
import json

if __name__ == '__main__':
    data = {}
    with open('nai.csv', newline='') as csvf:
        csvReader = csv.reader(csvf, delimiter=';')
        for row in csvReader:
            name = row[0]
            ratings = {}
            print(name)
            for i in range(1, len(row), 2):
                if row[i]:
                    ratings.update({row[i]: row[i+1]})
            data[name] = ratings

    with open('ratings.json', 'w', encoding='utf-8') as jsonf:
        jsonf.write(json.dumps(data, indent=4))

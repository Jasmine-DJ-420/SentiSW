# csv writing
import csv


class CsvOp:
    def __init__(self, header, path):
        self.header = header
        self.path = path

    def init_csv(self):
        with open(self.path, 'w') as file:
            writer = csv.DictWriter(file, self.header)
            writer.writeheader()

    def write_csv(self, line):
        with open(self.path, 'a') as file:
            writer = csv.DictWriter(file, self.header)
            writer.writerow(line)

    def write_csv_array(self, array):
        with open(self.path, 'a') as file:
            writer = csv.DictWriter(file, self.header)
            writer.writerows(array)

    def read_csv(self):
        with open(self.path, 'r') as file:
            reader = csv.DictReader(file)
            ret = []
            for line in reader:
                ret.append(line)
        return ret
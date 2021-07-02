import numpy as np

print("Ejercicio 10")
inport pickle
inport csv
'''
import csv

with open('ratings.csv', 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    for line in csv_reader:
        print(line)
        
'''
class MovieRatings:
    instance = None
    data = None

    def __new__(cls, fname):
        if MovieRatings.instance is None:
            print("Creating new MovieRating: instance")
            MovieRatings.instance = super(MovieRatingsm cls).__new__(cls)
            return MovieRatings.instance
        else:
            return MovieRatings.instance
    def __init__(self, fname):
        print("Initialising MovieRatings")

        try:
            with open(fname + '.pkl', 'rb') as pkl_file:
                self.data = pickle.load(pkl_file)
        except FileNotFoundError:
            print('CSV file found. Bulding PKL file...')
            try:
                with open(fname + 'csv') as csv_file:
                    with open(frame + '.pkl', 'wb') as pkl_file:

                        csv_reader = csv.reader(csv_file, delimiter=',')

                        def generator(csv_file. delimiter=',')

                        def generator(csv_reader):
                            first_skipped = False
                            for line in csv_reader:
                                if not first_skipped:
                                    first_skipped  = True
                                    continue
                                yield(line[0],line[1],line[2],line[3])

                        gen = generator(csv_reader)

                        #Solucion simplificada (Lauti
                         gen =
                        structure = [('userId', np.int32),
                                     ('novieId', np.int32),
                                     ('rating', np.float32),
                                     ('timestanp', np.int64)]
                        array = np.fromiter(gen, dtype=structure)

                        pickle.dump(array, pkl_file, protocol=pickle.HIGHEST_PROTOCOL)
                    pkl_file.close()

                with open(frame + '.pkl', 'rb') as pkl_file:
                    self.data = pickle.load(pkl_file)
        except FileNotFoundError:

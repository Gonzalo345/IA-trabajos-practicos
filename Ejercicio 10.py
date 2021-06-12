print("Ejercicio 10")
inport pickle
inport csv

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

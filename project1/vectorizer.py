import numpy as np

class Vectorizer:
    """
        Transform raw data into feature vectors. Support ordinal, numerical and categorical data.
        Also implements feature normalization and scaling.

        TODO: Support numerical, ordinal, categorical, histogram features.
    """
    def __init__(self, feature_config, num_bins=5):
        self.feature_config = feature_config
        self.feature_transforms = {}
        self.is_fit = False
        self.num_bins = num_bins 

    def get_numerical_vectorizer(self, values, verbose=False):
        """
        :return: function to map numerical x to a zero mean, unit std dev normalized score.
        """

        mean = np.mean(values)
        std = np.std(values)

        if verbose:
            print(f"Numerical vectorizer: mean={mean}, std={std}") 

        def vectorizer(x):
            """
            :param x: numerical value
            Return transformed score

            Hint: this fn knows mean and std from the outer scope
            """
            # Handle missing values
            if x is None or x == '':
                x = mean

            x = float(x)

            if std > 0:
                return (x - mean) / std
            else:
                return x - mean

        return vectorizer

    def get_histogram_vectorizer(self, values, verbose=False):
        """
        :return: function to map histogram x to a normalized histogram feature vector
        """
        all_bins = []
        for v in values:
            all_bins.extend(v)

        all_bins = [float(b) for b in all_bins]
        min_bin = min(all_bins)
        max_bin = max(all_bins)
        bin_edges = np.linspace(min_bin, max_bin, num=self.num_bins + 1)  # num_bins => num_bins + 1 edges

        if verbose:
            print(f"Histogram vectorizer: min={min_bin}, max={max_bin}, edges={bin_edges}")

        def vectorizer(x):
            """
            :param x: list of numerical values
            Return histogram feature vector
            """
            hist, _ = np.histogram(x, bins=bin_edges)
            hist = hist / np.sum(hist) if np.sum(hist) > 0 else hist
            return hist

        return vectorizer

    def get_categorical_vectorizer(self, values, verbose=False):
        """
        :return: function to map categorical x to one-hot feature vector
        """
        unique_vals = np.unique(values)
        val_to_index = {val: i for i, val in enumerate(unique_vals)}

        #print(f"Categorical vectorizer: unique_vals={unique_vals}")

        def vectorizer(x):
            # If x not in mapping, return zero vector (unknown category)
            if x is None or x == '':
                return [0] * len(unique_vals)

            vector = [0] * len(unique_vals)
            if x in val_to_index:
                vector[val_to_index[x]] = 1
            
            return vector
        
        return vectorizer

    def fit(self, X):
        """
            Leverage X to initialize all the feature vectorizers (e.g. compute means, std, etc)
            and store them in self.

            This implementation will depend on how you design your feature config.
        """

        for feature_type, feature_names in self.feature_config.items():
            for feature_name in feature_names:
                values = [x[feature_name] for x in X if x[feature_name] is not None and x[feature_name] != '']
                if feature_type == 'numerical':
                    self.feature_transforms[feature_name] = self.get_numerical_vectorizer(np.array(values).astype(float), verbose=False)
                elif feature_type == 'categorical':
                    self.feature_transforms[feature_name] = self.get_categorical_vectorizer(values, verbose=False)
                elif feature_type == 'histogram':
                    self.feature_transforms[feature_name] = self.get_histogram_vectorizer(values, verbose=False)
                else:
                    raise ValueError(f"Unknown feature type: {feature_type}")

        self.is_fit = True


    def transform(self, X):
        """
        For each data point, apply the feature transforms and concatenate the results into a single feature vector.

        :param X: list of dicts, each dict is a datapoint
        """

        if not self.is_fit:
            raise Exception("Vectorizer not intialized! You must first call fit with a training set" )

        transformed_data = []
        
        for x in X:
            feature_vector = []
            for feature_name, transform in self.feature_transforms.items():
                if feature_name in x and x[feature_name] is not None:
                    transformed_feature = transform(x[feature_name])
                    feature_vector.extend(transformed_feature if isinstance(transformed_feature, (list, np.ndarray)) else [transformed_feature])
                else:
                    feature_vector.extend([0] * len(transform(0)))  # Handle missing features
            transformed_data.append(feature_vector)

        return np.array(transformed_data)
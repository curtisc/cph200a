import numpy as np

class LogisticRegression():
    """
        A logistic regression model trained with stochastic gradient descent.
    """

    def __init__(self, num_epochs=100, learning_rate=1e-4, batch_size=16, regularization_lambda=0,  verbose=False):
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.verbose = verbose
        self.regularization_lambda = regularization_lambda

    def sigmoid(self, z):
        """
            Compute the sigmoid function.
        """
        # For numerical stability, use different formulas for positive and negative z
        return np.where(z >= 0, 
                        1 / (1 + np.exp(-z)),
                        np.exp(z) / (1 + np.exp(z)))
    
    def logistic_loss(self, y_true, y_pred, eps=1e-12):
        """
            Calculate the logistic loss of a prediction.
        """
        # Clip to avoid log(0)
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def fit(self, X, Y, X_val=None, Y_val=None):
        """
            Train the logistic regression model using stochastic gradient descent.
        """

        # Initialize parameters
        num_samples, num_features = X.shape
        self.theta = np.random.randn(num_features)
        self.bias = 0.0
        self.train_losses = []
        self.val_losses = []

        for epoch in range(self.num_epochs):
            if self.verbose:
                print(f"Epoch {epoch+1}/{self.num_epochs}")

            # Shuffle the data at the beginning of each epoch
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            X_shuffled = X[indices]
            Y_shuffled = Y[indices]

            for start_idx in range(0, num_samples, self.batch_size):
                end_idx = min(start_idx + self.batch_size, num_samples)
                X_batch = X_shuffled[start_idx:end_idx]
                Y_batch = Y_shuffled[start_idx:end_idx]

                # Compute gradients
                d_theta, d_bias = self.gradient(X_batch, Y_batch)

                # Update parameters
                self.theta -= self.learning_rate * d_theta
                self.bias -= self.learning_rate * d_bias

            # Compute and log training loss
            train_pred = self.predict_proba(X)
            train_loss = self.logistic_loss(Y, train_pred)
            self.train_losses.append(train_loss)

            # Compute and log validation loss if validation data provided
            if X_val is not None and Y_val is not None:
                val_pred = self.predict_proba(X_val)
                val_loss = self.logistic_loss(Y_val, val_pred)
                self.val_losses.append(val_loss)

        if self.verbose:
            print("Training complete.")

    def gradient(self, X, Y):
        """
            Compute the gradient of the loss with respect to theta and bias with L2 Regularization.
            Hint: Pay special attention to the numerical stability of your implementation.
        """

        # Compute predictions
        predictions = self.predict_proba(X)

        # Compute gradients
        errors = predictions - Y
        d_theta = X.T @ errors / X.shape[0] + self.regularization_lambda * self.theta
        d_bias = np.sum(errors) / X.shape[0]

        return d_theta, d_bias

    def predict_proba(self, X):
        """
            Predict the probability of lung cancer for each sample in X.
        """

        return self.sigmoid(X @ self.theta + self.bias)

    def predict(self, X, threshold=0.5):
        """
            Predict the if patient will develop lung cancer for each sample in X.
        """

        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)
import pandas
import numpy as np


#Return -1, 0 or 1 using the sign() mathematical function
def binary_classification(test_data, expected_test, features, expected_var, epochs = 1, bias = 1):
    try:
        #This variable will be returned on the final result
        test_result = 0

        #Learning rate variable:
        r = 0.1

        #Inicialize random generator.
        rng = np.random.default_rng()

        #Transform the dataframe in a array of vectors.
        x_data = features.to_numpy()

        #Transform the test data into an array of arrays.
        test_data = test_data.to_numpy()

        #Inicializing the weights array with random values.
        weights = rng.uniform(low = -0.1, high = 0.1, size = x_data.shape[1])
        weights = np.insert(weights, 0, 0)

        #Do the iteration in the entire data in the epochs numbers of time.
        for epoch in range(epochs):
            print(f'Running: {epoch + 1 }th epoch')
            #Iterate over the items, and update the weights vector if necessary.
            for x, y in zip(x_data, expected_var):

                #Insert bias, calculate the final output value.
                x = np.insert(x, 0, 0)
                output = np.sign(np.dot(x, weights) + bias)
                output = 1 if output > 0 else 0

                #Correct the weights if it's not correct.
                if y - output != 0:
                    first_part = np.multiply(x, (r * (y - output)))
                    weights = np.add(weights, first_part)

        #Test the data
        for x, w, y in zip(test_data, weights, expected_test):
            x = np.insert(x, 0, 0)
            output = np.sign(np.dot(x, weights))
            output = 1 if output > 0 else 0

            #Increment one for each successful result
            if output == y:
                test_result += 1

        return test_result if test_result != 0 else 0
    except Exception as e:
        raise e
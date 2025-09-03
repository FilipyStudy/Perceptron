import pandas
import numpy as np


#Return -1, 0 or 1 using the sign() mathematical function
def binary_classification(features, expected_var, epochs):
    try:
        #Learning rate variable:
        r = 0.1

        #Inicialize random generator.
        rng = np.random.default_rng()

        #Inicializing the weights array with random values.
        weights = rng.uniform(low = 0.5, high = 0.5, size = features.size[1])

        #Transform the dataframe in a array of vectors.
        x_data = features.to_numpy()
        #Do the iteration in the entire data in the epochs numbers of time.
        for epoch in range(epochs):

            #Iterate over the items, and update the weights vector if necessary.
            for x, y in zip(x_data, expected_var):

                #Insert bias, calculate the final output value.
                x = np.insert(x, 0, 0.5)
                weights = np.insert(weights, 0, 0.5)
                output = np.sign(np.dot(x, weights))
                output = 0 if output <= 0 else 1

                #Correct the weights if it's not correct.
                if y - output != 0:
                    first_part = np.multiply(x, (r * (y - output)))
                    weights = np.add(weights, first_part)

        return weights


    except Exception as e:
        raise e
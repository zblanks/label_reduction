import numpy as np
import os


def gen_data(mu, n, path, overlap_lvl):
    """Generates our synthetic data for experimentation

    Args:
        mu (ndarray, shape = [n_class, 2]): Matrix of mu values
        n (int): Total number of samples to generate
        path (str): Path to save the synthetic data
        overlap_lvl (str): The level of data overlap we have

    Returns:
        None: Simply saves the data to disk at the specified location
    """

    # Create a placeholder for our data
    n_class = mu.shape[0]
    data = np.empty(shape=(n, n_class))

    # We need to determine the number of samples per class
    if n % n_class == 0:
        samples_per_class = n // n_class
    else:
        print('Zach, WTF are you doing -- this is supposed to be a simple'
              'example; make the number of samples divisible by the number of '
              'classes so we do not have to do crazy shit')
        return

    # Loop through and create our data for each class
    count = 0
    for i in range(n_class):
        # Generate the data
        tmp_distn = np.random.multivariate_normal(mean=mu[i, :],
                                                  cov=np.eye(n_class),
                                                  size=samples_per_class)
        # Add the data to our overall matrix
        data[count:(samples_per_class + count), :] = tmp_distn
        count += samples_per_class

    # Save the result to disk
    np.savetxt(os.path.join(path, overlap_lvl + '_overlap.csv'), X=data,
               delimiter=',')


def gen_mu(overlap_lvl):
    """Generates a mu matrix given the number of classes and how much
    overlap we want with the data

    Args:
        overlap_lvl (str): How much overlap we want

    Returns:
        ndarray, shape = [n_class, 2]: mu matrix
    """

    # Go through each level of overlap and create the corresponding matrix
    if overlap_lvl == 'none':
        mu_1 = -2 * np.ones(shape=(1, 2))
        mu_2 = 2 * np.ones(shape=(1, 2))
        return np.concatenate((mu_1, mu_2), axis=0)
    elif overlap_lvl == 'little':
        mu_1 = -1.5 * np.ones((1, 2))
        mu_2 = 1.5 * np.ones((1, 2))
        return np.concatenate((mu_1, mu_2), axis=0)
    elif overlap_lvl == 'some':
        mu_1 = 1 * np.ones((1, 2))
        mu_2 = -1 * np.ones((1, 2))
        return np.concatenate((mu_1, mu_2), axis=0)
    else:
        mu_1 = .7 * np.ones((1, 2))
        mu_2 = -.7 * np.ones((1, 2))
        return np.concatenate((mu_1, mu_2), axis=0)


if __name__ == '__main__':
    np.random.seed(17)

    # Generate our synthetic data with various levels of overlap
    file_path = os.path.join('C:/Users/zqb0731/Documents/label_reduction',
                             'synthetic_experiments/data')

    # Almost no overlap
    gen_data(mu=gen_mu('none'), n=100, path=file_path, overlap_lvl='no')

    # Small overlap
    gen_data(mu=gen_mu('little'), n=100, path=file_path, overlap_lvl='little')

    # Some overlap
    gen_data(mu=gen_mu('some'), n=100, path=file_path, overlap_lvl='some')

    # Quite a bit
    gen_data(mu=gen_mu('lots'), n=100, path=file_path, overlap_lvl='high')

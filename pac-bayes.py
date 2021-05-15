import math
import torch
import pickle
import itertools
import numpy as np

from util.data import get_data, normalize_data
from util.kernel import sanitise, increment_kernel, complexity, invert_bound
from util.trainer import train_network

### Dependent variables
num_train_examples_list = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
random_labels_list = [True, False]
binary_digits_list = [True, False]
depth_list = [2, 7]
width_list = [5000, 10000]
seed_list = [0, 1, 2]

### Data hyperparameters
batch_size = 50

### Training hyperparameters
init_lr = 0.01
decay = 0.9

### Estimator hyperparameters
num_samples = 10**6
cuda = True
delta = 0.01

param_product = itertools.product( num_train_examples_list, random_labels_list, binary_digits_list, depth_list, width_list, seed_list )

for params in param_product:
    print('\n', params)
    num_train_examples, random_labels, binary_digits, depth, width, seed = params

    ### Set random seed
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

    ### Get data
    full_batch_train_loader, _, train_loader, test_loader = get_data( num_train_examples=num_train_examples,
                                                                      num_test_examples=None,
                                                                      batch_size=batch_size,
                                                                      random_labels=random_labels,
                                                                      binary_digits=binary_digits )

    ### Train network
    train_acc, test_acc, model = train_network( train_loader = train_loader,
                                                test_loader = test_loader,
                                                depth=depth,
                                                width=width,
                                                init_lr=init_lr, 
                                                decay=decay )

    print(f"Train acc: {train_acc[-1]}")
    print(f"Test acc: {test_acc}")

    ### Kernel arithmetic
    data, target = list(full_batch_train_loader)[0]
    if cuda: data, target = data.cuda(), target.cuda()
    data, target = normalize_data(data, target)

    c = target.float()
    sigma = sanitise(torch.mm(data, data.t()) / data.shape[1])
    assert ( sigma == sigma.t() ).all()
    n = sigma.shape[0]

    for _ in range(depth-1):
        sigma = increment_kernel(sigma)
        assert ( sigma == sigma.t() ).all()
        
    c0_1, c1 = complexity( sigma, c, num_samples )
    c0_2, c1 = complexity( sigma, c, num_samples )
    c0_3, c1 = complexity( sigma, c, num_samples )

    delta_term = math.log(2*n/delta)

    estimator_1 = invert_bound( (c0_1 + delta_term) / (n-1) )
    estimator_2 = invert_bound( (c0_2 + delta_term) / (n-1) )
    estimator_3 = invert_bound( (c0_3 + delta_term) / (n-1) )

    bound = invert_bound( (c1 + delta_term) / (n-1) )

    print(f"Estimator 1: {estimator_1}")
    print(f"Estimator 2: {estimator_2}")
    print(f"Estimator 3: {estimator_3}")
    print(f"Bound: {bound}")

    results = (train_acc, test_acc, estimator_1, estimator_2, estimator_3, bound)
    fname = 'logs/pac-bayes/' + str(params) + '.pickle'
    pickle.dump( results, open( fname, "wb" ) )

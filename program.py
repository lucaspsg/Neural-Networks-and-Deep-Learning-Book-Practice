import pickle

import network
import mnist_loader

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

with open('./net.pkl', 'wb') as outp:
    net = network.Network([784, 30, 10])
    net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
    pickle.dump(net, outp, pickle.HIGHEST_PROTOCOL)




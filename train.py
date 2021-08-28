from numpy import ndarray, random
from libs.neural_network import *

cats_train : ndarray = np.load("dataset/cats_train.npy")
cats_test : ndarray = np.load("dataset/cats_test.npy")

print(f"cats train shape = {cats_train.shape}")
print(f"cats test shape = {cats_test.shape}")

cats_train = [{"i": 0, "img": img} for img in cats_train]
cats_test = [{"i": 0, "img": img} for img in cats_test]

rabbits_train : ndarray = np.load("dataset/rabbits_train.npy")
rabbits_test : ndarray = np.load("dataset/rabbits_test.npy")

print(f"rabbit train shape = {rabbits_train.shape}")
print(f"rabbit test shape = {rabbits_test.shape}")

rabbits_train = [{"i": 1, "img": img} for img in rabbits_train]
rabbits_test = [{"i": 1, "img": img} for img in rabbits_test]


training_data = cats_train + rabbits_train
test_data = cats_test + rabbits_test


del cats_train
del rabbits_train

nn = NeuralNetwok(784, 16, 2, 2)
nn.set_activation_functions(leaky_relu, sigmoid)
nn.set_learning_rate(0.001)

# index 0 = cat
# index 1 = rabbit
outputs = [[1, 0], [0, 1]]

print("training......")

for i in range(0, 100000):
    random.shuffle(training_data)
    random.shuffle(test_data)

    total_right = 0

    for data in test_data:
        prediction_index = np.argmax(nn.forward(data["img"]))

        if prediction_index == data["i"]:
            total_right += 1

    print(f"accuracy: {(total_right / len(test_data)) * 100}%")


    for data in training_data:
        nn.train(data["img"],outputs[data["i"]])
    
    print(f"Epoch: {i + 1}")

    nn.save("model/model.npy")
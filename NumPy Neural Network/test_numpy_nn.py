import numpy_nn as nn
import numpy as np


def test_ReLU():
    ex1 = np.array([[1,6,3,-1,-9,45,0,-4]]).T
    assert not np.any(nn.ReLU(ex1) - np.array([[1,6,3,0,0,45,0,0]]).T)

def test_Sigmoid():
    ex1 = np.array(([-1, 0, 1]))
    dif = nn.Sigmoid(ex1) - np.array([0.26894142, 0.5, 0.73105858])
    out = np.abs(dif) > 0.00001
    assert not np.any(out)

def test_dSigmoid():
    ex1 = np.array(([-1, 0, 1])).T
    dif = nn.dSigmoid(ex1) - np.array([[0.19661193, 0, 0], [0, 0.25, 0], [0, 0, 0.19661193]])
    out = np.abs(dif) > 0.00001
    assert not np.any(out)

def test_Softmax():
    ex1 = np.array([[100, 100]])
    ex2 = np.array([[100, 0]])
    raise NotImplementedError

def test_forward_pass():
    in1 = np.array([[83, -27]]).T
    net1 = [{'W': np.array([[1, 0], [0, 1]]), 'activation': nn.Identity, 'offset': np.array([[4], [9]])}]
    dif1 = np.abs((nn.forward_pass(net1, in1) - np.array([[87, -18]]).T))
    assert not np.any(dif1 > 0.00001)


    in2 = np.array([[83, -27]]).T
    net2 = [{'W': np.array([[1, 0], [0, 1]]), 'activation': nn.ReLU, 'offset': np.array([[0], [0]])}]
    dif2 = np.abs((nn.forward_pass(net2, in2) - np.array([[83, 0]]).T))
    assert not np.any(dif2 > 0.00001)


    in3 = np.array([[2, 1]]).T
    net3 = [{'W': np.array([[3, 2], [-1, 1]]), 'activation': nn.Identity, 'offset': np.array([[0], [1]])}]
    dif3 = np.abs((nn.forward_pass(net3, in3) - np.array([[5, 6]]).T))
    assert not np.any(dif3 > 0.00001)

    # two layer test:
    net4 = [{'W': np.array([[3, 2], [-1, 1]]), 'activation': nn.Identity, 'offset': np.array([[0], [0]])}, 
            {'W': np.array([[1, -1.1], [1, 1]]), 'activation': nn.ReLU, 'offset': np.array([[0], [0]])}]
    
    in41 = np.array([[2, 1]]).T
    dif41 = np.abs((nn.forward_pass(net4, in41) - np.array([[10, 0]]).T))
    assert not np.any(dif41 > 0.00001)

    in42 = np.array([[1, 1]]).T
    dif42 = np.abs((nn.forward_pass(net4, in42) - np.array([[5, 0.8]]).T))
    assert not np.any(dif42 > 0.00001)







if __name__ == "__main__":
    test_ReLU()
    test_Sigmoid()
    test_dSigmoid()
    # test_Softmax()
    test_forward_pass()
import unittest
import torch
import torchlazy as tl

def linear(x, out_features):
    linear_mod = tl.create(torch.nn.Linear, x.shape[-1], out_features)

    return linear_mod(x)

def mlp(x, hidden_features):
    x = linear(x, hidden_features)
    x = linear(x, hidden_features)
    x = linear(x, hidden_features)
    return x


class TorchlazyTest(unittest.TestCase):
    def test_build(self):
        inp = torch.randn(8, 3)
        mod = tl.build(mlp, inp, hidden_features = 5)

        mod(inp)
        mod(inp)
        print(mod.state_dict())

if __name__ == '__main__':
    unittest.main()

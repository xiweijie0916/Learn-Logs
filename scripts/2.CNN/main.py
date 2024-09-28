import torch
from cnn import SimpleCNN

if __name__ == "__main__":
    model = SimpleCNN()
    input = torch.randn(size=(4, 3, 28, 28))
    output = model(input)
    print(output)
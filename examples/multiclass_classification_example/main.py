import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import nn_library as nn
from fullyconnected_model import MultiLayerPerceptron

preprocessor = nn.Preprocessor(nn.Normalize())
train_data, test_data = nn.Datasets(test_split=0.2, val_split=None, Preprocessor=preprocessor).iris()
train_loader = nn.DataLoader(train_data, batch_size=8, shuffle=True)
test_loader = nn.DataLoader(test_data, batch_size=1, shuffle=False)

model = MultiLayerPerceptron()


def train(model, train_loader, epoch, log_interval=2):
    """
    A utility function that performs a basic training loop.

    For each batch in the training set, fetched using `train_loader`:
        - Zeroes the gradient used by `optimizer`
        - Performs forward pass through `model` on the given batch
        - Computes loss on batch
        - Performs backward pass
        - `optimizer` updates model parameters using computed gradient

    Prints the training loss on the current batch every `log_interval` batches.
    """
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # Performs forward pass through `model` on the given batch;
        # equivalent to model.forward(inputs)
        outputs = model(inputs)
        # Computes loss on batch;
        loss = model.loss_function(outputs, targets)
        # Performs backward pass; steps backward through the computation graph,
        # computing the gradient of the loss wrt model parameters.
        grad_z = model.loss_function.backward()
        model.backward(grad_z)
        # updates model parameters using computed gradient.
        model.step()
        # Prints the training loss on the current batch every `log_interval`
        # batches.
        if batch_idx % log_interval == 0:
            print(f"Train Epoch: {epoch+1} Batch: {batch_idx+1} Loss: {loss}")


def test(model, test_loader):
    """
    A utility function to compute the loss and accuracy on a test set by
    iterating through the test set using the provided `test_loader` and
    accumulating the loss and accuracy on each batch.
    """
    test_loss = 0.0
    correct = 0

    for inputs, targets in test_loader:
        # Performs forward pass through `model`
        outputs = model(inputs)
        
        # We use `reduction="sum"` to aggregate losses across batches using
        # summation instead of taking the mean - we will take the mean at
        # the end once we have accumulated all the losses.
        test_loss += model.loss_function(outputs, targets).sum()
        pred = outputs.argmax(axis=1).squeeze()
        
        targets = targets.argmax(axis=1).squeeze()
        correct += (pred == targets).sum()

    test_loss /= len(test_loader.dataset[0])
    accuracy = (correct / len(test_loader.dataset[0]))*100
    print(f"Test set: Average loss: {test_loss}, Accuracy: {accuracy}%\n")



if __name__ == "__main__":

    params = {"epochs": 200,
              "test interval": 1,
            }
    # test accuracy of random model
    test(model, test_loader)
    # run training loop
    for epoch in range(params["epochs"]):
        train(model, train_loader, epoch)
        if epoch % params["test interval"] == 0:
            test(model, test_loader)

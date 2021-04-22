
# %%
# Example from https://pytorch.org/tutorials/beginner/examples_nn/two_layer_net_module.html

import torch

class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred


# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
# N, D_in, H, D_out = 64, 1000, 100, 10
# N, D_in, H, D_out = 100, 2, 1000, 2
N, D_in, H, D_out = 100, 1, 100, 1

# Create random Tensors to hold inputs and outputs
def gt_func(in_vec):
    scale = 10 # 1000, 1
    return torch.floor(scale * in_vec) / scale
    # return in_vec

x = torch.randn(N, D_in)
y = gt_func(x)
x_test = torch.randn(N, D_in)
y_test = gt_func(x_test)

test_range = [-10, 10] #[-3, 3], [-5, 5]
x_gt = torch.linspace(test_range[0], test_range[1], 10000)
y_initial = model(x_gt.view([-1,1])).detach().numpy()

# x = torch.randn(N, D_in)
# y = torch.randn(N, D_out)
# x_test = torch.randn(N, D_in)
# y_test = torch.randn(N, D_out)

# Construct our model by instantiating the class defined above
model = TwoLayerNet(D_in, H, D_out)

# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
for t in range(500):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x)
    y_validation = model(x_test)

    # Compute and print loss
    loss = criterion(y_pred, y)
    loss_validation = criterion(y_validation, y_test)

    if t % 100 == 99:
        print("Training: ", t, loss.item())
        print("Validation:", t, loss_validation.item())

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# %%
print("***Parameters***")
for p in model.linear1.parameters():
    plt.hist(p.detach().numpy(), alpha=0.5)

# %%
test_range = [-10, 10] #[-3, 3], [-5, 5]
fig = plt.figure(figsize=(10, 10))
t = torch.linspace(test_range[0], test_range[1], 10000)
y_gt = gt_func(t)
plt.plot(t, y_gt, "m-")

# x_gt = torch.vstack((t,t)).T
# y_estimated = model(x_gt).detach().numpy()
# plt.plot(t, y_estimated[:,0], "b--")
# plt.plot(t, y_estimated[:,1], "k--")

y_estimated = model(x_gt.view([-1,1])).detach().numpy()
plt.plot(x_gt, y_estimated, "b--")
plt.plot(x_gt, y_initial, "r--")

# plt.scatter(x[:,0], y[:,0], c="r", marker="x")
# plt.scatter(x_test[:,0], y_test[:,0], c="b", marker="o")
plt.xlim(test_range)
plt.ylim(test_range)


# %%

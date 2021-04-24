
# %%
# Example from https://pytorch.org/tutorials/beginner/examples_nn/two_layer_net_module.html

import math
import matplotlib
import matplotlib.pyplot as plt
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



class ThreeLayerNet(torch.nn.Module):
    def __init__(self, D_in, H1, H2, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(ThreeLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H1)
        self.linear2 = torch.nn.Linear(H1, H2)
        self.linear3 = torch.nn.Linear(H2, D_out)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        h1_relu = self.linear1(x).clamp(min=0)
        h2_relu = self.linear2(h1_relu).clamp(min=0)
        y_pred  = self.linear3(h2_relu)
        return y_pred


# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
# N, D_in, H, D_out = 64, 1000, 100, 10
# N, D_in, H, D_out = 100, 2, 1000, 2
N, D_in, H1, H2, D_out = 100, 1, 100, 100, 1

# Construct our model by instantiating the class defined above
model2 = TwoLayerNet(D_in, H1, D_out)
model3 = ThreeLayerNet(D_in, H1, H2, D_out)
model = model3

# Create random Tensors to hold inputs and outputs
def gt_func(in_vec):
    scale = 10 # 1000, 1
    # return torch.floor(scale * in_vec) / scale
    # return torch.floor(scale * in_vec**3) / scale
    # return in_vec**3
    return torch.abs(in_vec)


x = torch.randn(N, D_in)
y = gt_func(x)
x_test = torch.randn(N, D_in)
y_test = gt_func(x_test)

num_gt_points = 10000
test_range = [-10, 10] #[-3, 3], [-5, 5]
x_gt = torch.linspace(test_range[0], test_range[1], num_gt_points)
y_initial = model(x_gt.view([-1,1])).detach().numpy()


# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
num_iterations = 500
evaluations = torch.zeros(num_iterations, num_gt_points)
for t in range(num_iterations):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x)
    y_validation = model(x_test)

    # Compute and print loss
    loss = criterion(y_pred, y)
    loss_validation = criterion(y_validation, y_test)

    if t % 100 == 99:
        print("Training: ", t, loss.item())
        print("Validation:", t, loss_validation.item())

    evaluations[t,:] = model(x_gt.view([-1,1])).view([1,-1])

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


fig = plt.figure(figsize=(10, 10))

y_gt = gt_func(x_gt)
y_estimated = model(x_gt.view([-1,1])).detach().numpy()

plt.plot(x_gt, y_gt, "m-", label="gt")
plt.plot(x_gt, y_estimated, "b--", label="estimated")
plt.plot(x_gt, y_initial, "r--", label="initial")

num_plots = 10
for t in range(1, num_iterations, math.floor(num_iterations/num_plots)):
    plt.plot(x_gt, evaluations[t,:].detach().numpy(), "k--", alpha=(1 - (t+1)/num_iterations))

plot_range = [-5, 5] # test_range, [-3, 3], [-5, 5]
plt.xlim(plot_range)
plt.ylim(plot_range)
plt.legend()


# %%
print("***Parameters***")
for p in model.linear1.parameters():
    plt.hist(p.detach().numpy(), alpha=0.5)

# %%

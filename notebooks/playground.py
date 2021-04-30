
# %%
# Example from https://pytorch.org/tutorials/beginner/examples_nn/two_layer_net_module.html

import math
import matplotlib
import matplotlib.pyplot as plt
import numpy
import torch

torch.manual_seed(271828)

###
# Set options.
should_use_derivatives = False
num_iterations = 500 # 500, 5000
criterion = torch.nn.MSELoss(reduction='sum') # torch.nn.L1Loss(reduction='sum')

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H1, H2, D_out = 100, 1, 100, 100, 1


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


def compute_derivative_1d(func, t):
    y = func(t)
    m = y.numel()
    n = t.numel()

    basis = torch.eye(m, dtype=y.dtype)
    dy_dt = torch.empty(m, n, dtype=y.dtype)
    for i in range(m):
        gradient = torch.autograd.grad(y[i], t, retain_graph=True, create_graph=True)
        dy_dt[i, :] = gradient[0].T

        # # y[i].backward(retain_graph=True)
        # y[i].backward(create_graph=True, retain_graph=True)
        # gradient = t.grad
        # dy_dt[i, :] = gradient[:,0]
        # t.grad.data.zero_()

    derivative = torch.diag(dy_dt)
    return derivative


# Construct our model by instantiating the class defined above
model2 = TwoLayerNet(D_in, H1, D_out)
model3 = ThreeLayerNet(D_in, H1, H2, D_out)
model = model3

# Create random Tensors to hold inputs and outputs
def gt_func(in_vec):
    # return 2 * in_vec

    # scale = 10 # 1000, 1
    # return torch.floor(scale * in_vec) / scale
    # return torch.floor(scale * in_vec**3) / scale
    # return in_vec**3
    # return torch.abs(in_vec)

    num = in_vec.shape[0]
    poly = torch.zeros(num)
    
    # poly_scalars = torch.tensor([-1.0, 0.0, 1.0])
    # poly_scalars = 0.1 * torch.tensor([-1.0, 3.0, -4.5, -6.4, 2.4])
    poly_scalars = 0.1 * torch.tensor([-1.0, 3.0, -4.5, -6.4, 2.4, -3.0, 5.0, -0.1])
    deg = poly_scalars.shape[0]

    for ii in range(deg):
        poly = poly + poly_scalars[ii]*torch.squeeze(torch.pow(in_vec, ii))

    return poly.view([-1,1])

###
# Define inputs, outputs.
x = torch.randn(N, D_in, requires_grad=True)
y = gt_func(x)
y_derivative = compute_derivative_1d(gt_func, x)

# Verification set.
x_test = torch.randn(N, D_in, requires_grad=True)
y_test = gt_func(x_test)
y_derivative_test = compute_derivative_1d(gt_func, x_test)

# GT for plotting.
num_gt_points = 1000
test_range = [-10, 10] #[-3, 3], [-5, 5]
x_gt = torch.linspace(test_range[0], test_range[1], num_gt_points)
y_initial = model(x_gt.view([-1,1])).detach().numpy()

# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
if should_use_derivatives:
    print ("*** Optimize using function values and derivatives")
else:
    print ("*** Optimize using only function values")

optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
evaluations = torch.zeros(num_iterations, num_gt_points) # Used for plotting iterations.
for t in range(num_iterations):
    # Forward pass: Compute predicted y by passing x to the model
    if not should_use_derivatives:
        y_pred = model(x)
        y_validation = model(x_test)
    elif should_use_derivatives:
        y_pred_derivative = compute_derivative_1d(model, x)
        y_validation_derivative = compute_derivative_1d(model, x_test)

    # Compute and print loss
    if not should_use_derivatives:
        loss = criterion(y_pred, y)
        loss_validation = criterion(y_validation, y_test) 
    elif should_use_derivatives:
        # T0 = torch.tensor(0, dtype=torch.float32)
        # loss = criterion(y_pred_derivative, y_derivative) + criterion(model(T0.view([-1,1])), gt_func(T0.view([1,1])))
        # loss_validation = criterion(y_validation_derivative, y_derivative_test) + criterion(model(T0.view([-1,1])), gt_func(T0.view([1,1])))

        T0 = torch.tensor(0, dtype=torch.float32)
        loss = \
            criterion(y_pred_derivative, y_derivative)\
            +  torch.tensor(N, dtype=torch.float32) * criterion(model(T0.view([-1,1])), gt_func(T0.view([1,1])))
        loss_validation =\
            criterion(y_validation_derivative, y_derivative_test)\
            + torch.tensor(N, dtype=torch.float32) * criterion(model(T0.view([-1,1])), gt_func(T0.view([1,1])))

    # Track training and validation error.
    print_every_other = 50
    if t % print_every_other == 0:
        print("Training: ", t, loss.item())
        print("Validation:", t, loss_validation.item())
    evaluations[t,:] = model(x_gt.view([-1,1])).view([1,-1])

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()
print ("******* DONE OPTIMIZATION *******")

###
# Plot functions
fig = plt.figure(figsize=(10, 10))

y_gt = gt_func(x_gt)
y_estimated = model(x_gt.view([-1,1])).detach().numpy()

plt.scatter(x.detach().numpy(), y.detach().numpy(), c="m", marker="x", label="sampled points")
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
plt.grid(which="both")

# %%

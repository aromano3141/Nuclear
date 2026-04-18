import torch
import torch.nn as nn

class PINN(nn.Module):
    def __init__(self, in_features=6, hidden_dim=64, out_features=1):
        super(PINN, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, out_features)
        )

    def forward(self, x):
        # x is expected to be [batch, 6] representing [x, y, z, t, ndvi, soil_moisture]
        return self.net(x)

def advection_diffusion_reaction_loss(model, inputs):
    """
    Computes the physics-informed loss for the Advection-Diffusion-Reaction equation.
    inputs: Tensor of shape [batch, 6], where columns are [x, y, z, t, ndvi, soil_moisture]
            Requires requires_grad=True
    """
    inputs.requires_grad_(True)

    # Forward pass to get concentration prediction C
    C = model(inputs)

    # Compute gradients of C with respect to inputs
    # grad_outputs must be ones to sum up the gradients
    grads = torch.autograd.grad(
        C, inputs,
        grad_outputs=torch.ones_like(C),
        create_graph=True,
        retain_graph=True
    )[0]

    # Extract first derivatives
    dC_dx = grads[:, 0:1]
    dC_dy = grads[:, 1:2]
    dC_dz = grads[:, 2:3]
    dC_dt = grads[:, 3:4]

    # Compute second derivatives for spatial dimensions (diffusion)
    # We need to compute gradients of dC_dx with respect to x, etc.
    d2C_dx2 = torch.autograd.grad(
        dC_dx, inputs,
        grad_outputs=torch.ones_like(dC_dx),
        create_graph=True,
        retain_graph=True
    )[0][:, 0:1]

    d2C_dy2 = torch.autograd.grad(
        dC_dy, inputs,
        grad_outputs=torch.ones_like(dC_dy),
        create_graph=True,
        retain_graph=True
    )[0][:, 1:2]

    d2C_dz2 = torch.autograd.grad(
        dC_dz, inputs,
        grad_outputs=torch.ones_like(dC_dz),
        create_graph=True,
        retain_graph=True
    )[0][:, 2:3]

    # Placeholder physics parameters (these could be learnable or fixed)
    # Velocity components (advection)
    v_x, v_y, v_z = 0.1, 0.1, -0.05

    # Diffusion coefficient
    D = 0.01

    # Reaction/Decay rate (e.g., for 137Cs)
    lambda_decay = 0.02

    # Advection-Diffusion-Reaction Equation Residual:
    # dC/dt + v \cdot \nabla C - D \nabla^2 C + \lambda C = 0
    advection = v_x * dC_dx + v_y * dC_dy + v_z * dC_dz
    diffusion = D * (d2C_dx2 + d2C_dy2 + d2C_dz2)
    reaction = lambda_decay * C

    residual = dC_dt + advection - diffusion + reaction

    # Physics loss is the mean squared error of the residual
    physics_loss = torch.mean(residual ** 2)

    return physics_loss

if __name__ == "__main__":
    # Test instantiation and physics loss computation
    model = PINN()
    # Batch size 10, 6 features [x, y, z, t, ndvi, soil_moisture]
    dummy_inputs = torch.rand(10, 6)

    # Calculate loss
    phys_loss = advection_diffusion_reaction_loss(model, dummy_inputs)

    print("Nuclear_2 PINN model instantiated.")
    print("Computed Physics Loss:", phys_loss.item())

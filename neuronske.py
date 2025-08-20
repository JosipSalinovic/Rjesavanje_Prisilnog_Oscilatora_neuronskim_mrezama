import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

def exact_solution(d, w0, t):
    #Analitičko rješenje prisilinog gušenoga oscilatora
    assert d < w0
    wg = np.sqrt(w0**2 - d**2) 
    # frekvencija gusenja, d koeficijent gusenja gamma 
    A = -50/(3*np.sqrt(11))
    hom = torch.sin(wg*t)
    part=5*torch.sin(w0*t)
    exp = torch.exp(-d*t)
    u = exp*A*hom+part
    return u


class FCN(nn.Module):
    #definira potpuno spojenu neuronsku mrežu

    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS):
        super().__init__()
        activation = nn.Tanh
        self.fcs = nn.Sequential(*[
            nn.Linear(N_INPUT, N_HIDDEN),
            activation()])
        self.fch = nn.Sequential(*[
                     nn.Sequential(*[
                     nn.Linear(N_HIDDEN, N_HIDDEN),
                     activation()]) for _ in range(N_LAYERS - 1)]) 
        self.fce = nn.Linear(N_HIDDEN, N_OUTPUT)

    def forward(self, x):
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)
        return x


# stvori neuronsku mrežu
pinn = FCN(1, 1, 120, 3)

# definiraj početne točke treniranja
t_boundary = torch.tensor(0.).view(-1, 1).requires_grad_(True)

# definiraj točke treniranja za domenu diferencijalne jed. prisilnog oscilatora
t_physics = torch.linspace(0, 4, 120).view(-1, 1).requires_grad_(True)

# treniraj mrežu
mu, k = 1, 50
d, w0 = mu/(2*0.5), 10
t_test = torch.linspace(0, 4, 800).view(-1, 1)
u_exact = exact_solution(d, w0, t_test)
optimiser = torch.optim.Adam(pinn.parameters(), lr=1e-3)

for i in range(20001):
    optimiser.zero_grad()

    # hiperparametri regulacije članova
    lambda1, lambda2 = 1e-1, 1e-3

    # računaj  grešku ili loss za početnu točku
    u = pinn(t_boundary)  # (1,1)
    loss1 = (torch.squeeze(u))**2

    dudt = torch.autograd.grad(u, t_boundary, torch.ones_like(u), create_graph=True)[0]
    loss2 = (torch.squeeze(dudt))**2

    # računaj loss za fiziklni član (diferencijalnu jed. prisilnog oscilatora)
    u = pinn(t_physics)
    dudt = torch.autograd.grad(u, t_physics, torch.ones_like(u), create_graph=True)[0]
    d2udt2 = torch.autograd.grad(dudt, t_physics, torch.ones_like(dudt), create_graph=True)[0]
    force=-50*torch.cos(w0*t_physics)
    loss3 = torch.mean((0.5*d2udt2 + mu*dudt + k*u+force)**2)

    # algoritam backpropagation ili gradijentni spust (ažuriraj težine za svaku vezu) i optimiziraj
    loss = loss1 + lambda1*loss2 + lambda2*loss3
    loss.backward()
    optimiser.step()

    # plotaj graf i prati kako treniranje napreduje i greška se minimizira
    if i % 5000 == 0:
        u = pinn(t_test).detach()
        plt.figure(figsize=(6, 2.5))
        plt.scatter(t_physics.detach()[:, 0],torch.zeros_like(t_physics)[:, 0],s=20, lw=0, color="tab:green", alpha=0.6)
        
        plt.scatter(t_boundary.detach()[:, 0],torch.zeros_like(t_boundary)[:, 0],s=20, lw=0, color="tab:red", alpha=0.6)
        
        plt.plot(t_test[:, 0], u_exact[:, 0], label="Egzaktno rješenje", color="tab:grey", alpha=0.6)
        plt.plot(t_test[:, 0], u[:, 0], label="Neuronsko rješenje", color="tab:green")
        plt.title(f"Trening broj: {i}")
        plt.legend()
        plt.show()
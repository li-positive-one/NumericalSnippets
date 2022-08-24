import torch

def P2C(rho, u, p, gamma=1.4):
    return rho, rho*u, p/(gamma-1)+0.5*rho*u**2

def C2P(rho, mu, E, gamma=1.4):
    return rho,mu/rho,(gamma-1)*(E-0.5*mu**2/rho)

def eulerflux(F, gamma=1.4):
    rho=F[...,0]
    u=F[...,1]/rho
    E=F[...,2]
    p=(gamma-1)*(E-0.5*rho*u*u)
    flux = torch.stack([rho*u,rho*u**2+p,u*(E+p)],-1)
    return flux

def soundspeed(rho,p,gamma=1.4):
    return (gamma*p/rho)**0.5
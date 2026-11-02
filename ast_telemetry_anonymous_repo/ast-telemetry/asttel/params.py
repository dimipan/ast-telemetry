from dataclasses import dataclass


@dataclass(frozen=True)
class Theta:
    alpha: float = 0.5
    beta: float = 0.5
    tau_g: float = 0.01
    upsilon_star: float = 0.5
    n: int = 4
    lam: float = 5.0
    d_min: float = 0.2
    eps_p: float = 1e-9
    eps_g: float = 1e-12


THETA = Theta()
THETA_SI = 0.20
UPSILON_COMPLETE = 0.80

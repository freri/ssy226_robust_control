import casadi as cs
from pathlib import Path
import json, os
import numpy as np
class ATRModel():
    def __init__(self):
        config_path = Path(__file__)
        config_path = os.path.join(str(config_path.parent.parent), 'config', 'config.json')
        config_path = config_path.replace('/build/', '/src/')
        with open(config_path) as f: 
            config = json.load(f)
        # Base config
        base = config['base']
        self.N = base['N'] # The MPC horizon length
        self.NX = base['NX']  # The number of elements in the state vector
        self.NU = base['NU']  # The number of elements in the control vector
        self.ts = base['atr_ts']
        self.id = base['atr_id']

    def dynamics_ct(self, _x, _u):
        return cs.vcat([_u[0] * cs.cos(_x[2]),
                        _u[0] * cs.sin(_x[2]),
                        _u[1]])

    def dynamics_dt(self, x, u):
        dx = self.dynamics_ct(x, u)
        return cs.vcat([x[i] + self.ts * dx[i] for i in range(self.NX)])

    def dynamics_dt_rk4(self, _x, _u):
        f = self.dynamics_ct(_x, _u)
        k1 = f * self.ts
        f = self.dynamics_ct(_x + 0.5*k1, _u)
        k2 = f * self.ts
        f = self.dynamics_ct(_x + 0.5*k2, _u)
        k3 = f * self.ts
        f = self.dynamics_ct(_x + k3, _u)
        k4 = f * self.ts            

        return cs.vcat([_x[i] + (1/6) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) for i in range(self.NX)])
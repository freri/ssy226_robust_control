from os import name

import casadi as cs
import opengen as og

import os
from pathlib import Path
import json
try:
    from .atr_model import ATRModel
except ImportError:
    from atr_model import ATRModel


class MPC():
    def __init__(self):
        self.init_params()
        self.atr = ATRModel()

    def build(self):
        # Symbolics
        x_0 = cs.SX.sym('x_0', self.NX)
        x_ref = cs.SX.sym('x_ref', self.NX)
        u_ref = cs.SX.sym('u_ref', self.NU * self.N)
        u_k = cs.SX.sym('u_k', self.N * self.NU)
        u_0 = cs.SX.sym('u_0', self.NU)
        traj = cs.SX.sym('atr_traj', self.NX * self.N)

        x_k = x_0
        total_cost = 0

        for k in range(0, self.N):        
            x_ref = traj[(k)*self.NX:(k+1)*self.NX]
            x_k = self.atr.dynamics_dt_rk4(
                x_k, u_k[k * self.NU:(k+1) * self.NU])  # update state
            total_cost += self.stage_cost(x_k, u_k[k * self.NU:(
                k+1) * self.NU], x_ref, u_ref[k * self.NU:(k+1) * self.NU])
            
        total_cost += self.acc_cost(u_k, u_0)
        # total_cost += self.negative_v_cost(u_k)
        total_cost += self.soft_control_constraints(u_k)
        # total_cost += self.terminal_cost(x_k, x_ref)  # terminal cost
        vars = []

        vars += [u_k]
        vars = cs.vertcat(*vars)

        params = []
        params += [x_0]
        params += [u_0]
        params += [traj]
        params += [u_ref]

        params = cs.vertcat(*params)

        umin = [-0.5, -3.0] * self.N
        umax = [1.5, 3.0] * self.N


        bounds = og.constraints.Rectangle(umin, umax)

        # f2 = cs.fmax(0.0, cs.fabs(u_k[0::2] / umax[0]) + cs.fabs(0.5*u_k[1::2] / umax[0]) - 1)
    
        problem = og.builder.Problem(vars,
                                     params,
                                     total_cost) \
                                     .with_constraints(bounds) \
                                     #.with_penalty_constraints(f2)
                                    
            

        build_config = og.config.BuildConfiguration()\
            .with_build_directory(self.build_directory) \
            .with_build_mode(self.build_mode) \
            .with_build_python_bindings() \

        meta = og.config.OptimizerMeta()\
            .with_optimizer_name(self.optimizer_name)

        
        solver_config = og.config.SolverConfiguration()\
            .with_tolerance(1e-4) \
            .with_max_duration_micros(self.solver_max_duration_micros)
            # .with_delta_tolerance(1e-3)

        builder = og.builder.OpEnOptimizerBuilder(problem,
                                                  meta,
                                                  build_config,
                                                  solver_config)

        buildpath = Path(__file__)
        os.chdir(buildpath.parent)
        builder.build()

    def soft_control_constraints(self, u_k):
        constraint_violation = cs.fmax(0.0, cs.fabs(u_k[0::2]/1.5) + cs.fabs(0.5*u_k[1::2]/1.5) - 1 )
        return cs.mtimes([constraint_violation.T, self.q_u, constraint_violation])

    def stage_cost(self, _x, _u, _x_ref, _u_ref=None):
        if _u_ref is None:
            _u_ref = cs.DM.zeros(_u.shape)
        dx = _x - _x_ref
        du = _u - _u_ref
        return cs.mtimes([du.T, self.R, du]) + cs.mtimes([dx.T, self.Q, dx])

    # The terminal cost for x
    def terminal_cost(self, _x, _x_ref):
        dx = _x - _x_ref
        return cs.mtimes([dx.T, self.QN, dx])

    def acc_cost(self, _u, _u0):
        v = _u[0::2]
        w = _u[1::2]
        dv = (v - cs.vertcat(_u0[0], v[0:-1])) / self.atr_ts
        dw = (w - cs.vertcat(_u0[1], w[0:-1])) / self.atr_ts
        return cs.mtimes(dv.T, dv) * self.q_dv + cs.mtimes(dw.T, dw) * self.q_dw

    def negative_v_cost(self, _u):
        v_neg = cs.fmin(0, _u[0::2])
        return cs.mtimes(v_neg.T, v_neg) * self.q_vneg

    def init_params(self):
        config_path = Path(__file__)
        config_path = os.path.join(
            str(config_path.parent.parent), 'config', 'config.json')
        config_path = config_path.replace('/build/', '/src/')
        with open(config_path) as f:
            config = json.load(f)

        base = config['base']
        weights = config['mpc_weights']

        # Base config
        self.N = base['N']  # The MPC horizon length
        self.NX = base['NX']  # The number of elements in the state vector
        self.NU = base['NU']  # The number of elements in the control vector
        self.atr_ts = base['atr_ts']
        self.solver_max_duration_micros = base['solver_max_duration_micros']

        # Build config
        self.build_directory = base['build_directory']
        self.build_mode = base['build_mode']
        self.optimizer_name = base['optimizer_name']

        # Weights
        q_xy = weights['q_xy'] # Position error
        q_theta = weights['q_theta'] # Angle error
        qN_xy = weights['qN_xy'] # Terminal position error
        qN_theta = weights['qN_theta'] # Terminal angle error
        q_v = weights['q_v'] # Linear velocity 
        q_w = weights['q_w'] # Angular velocity 
        self.q_dv = weights['q_dv'] # Linear acceleration
        self.q_dw = weights['q_dw'] # Angular acceleration
        self.q_vneg = weights['q_vneg'] # Negative linear velocity
        self.q_u = weights['q_u'] # Soft constraint for control space of u

        self.Q = cs.SX.eye(self.NX) * [q_xy, q_xy, q_theta]
        self.R = cs.SX.eye(self.NU) * [q_v, q_w]
        self.QN = cs.SX.eye(self.NX) * [qN_xy, qN_xy, qN_theta]


if __name__ == '__main__':
    MPC().build()

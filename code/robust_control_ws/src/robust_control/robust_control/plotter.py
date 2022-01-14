import numpy as np
import time
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np
import pyqtgraph as pg
from pathlib import Path
import os
import json
import sys
from scipy.spatial.distance import cdist
class Plotter(QtGui.QMainWindow):
    def __init__(self, saved_trajectories, saved_traj_refs, saved_states, saved_u, saved_vel_refs, new_traj_idx, traj_len, logger, details=False, parent=None):
        super(Plotter, self).__init__(parent)
        config_path = Path(__file__)
        config_path = os.path.join(
            str(config_path.parent.parent), 'config', 'config.json')
        config_path = config_path.replace('/build/', '/src/')
        with open(config_path) as f:
            config = json.load(f)
        simulation = config['simulation']
        base = config['base']
        self.traj_update_rate = simulation["traj_update_rate"] #in seconds
        self.traj_len = traj_len
        self.N = base['N']  # The MPC horizon length
        self.NX = base['NX']  # The number of elements in the state vector
        self.NU = base['NU']  # The number of elements in the control vector
        self.ts = base['atr_ts']  # Controller sampling time
        self.traj_ts = base['traj_ts'] 
        self.atr_id = base['atr_id']
        self.logger = logger
        self.new_traj_idx = new_traj_idx
        # If false, plots only the trajectories
        self.plot_details = details
        # Number of states simulated
        self.Nsim = int(len(saved_states)/self.NX)
        self.sim_time = self.Nsim * self.ts
        # Length of all the trajectories (not splined)        
        self.traj_len_all = int(len(saved_trajectories)/self.NX)

        # # Number of trajectories
        self.N_traj = int(self.traj_len_all / self.traj_len)
        
        self.active_traj = -1

        # Data
        self.trajectories = np.reshape(
            saved_trajectories, (self.traj_len_all, self.NX))[:, :2]
        
        self.saved_traj_refs = np.reshape(
            saved_traj_refs, (self.Nsim, self.NX))[:, :2]

        self.states = np.reshape(saved_states, (self.Nsim, self.NX))

        if self.plot_details:
            self.lin_vel = saved_u[0::2]
            self.ref_lin_vel = saved_vel_refs[0::2]
            self.ang_vel = saved_u[1::2]
            self.ref_ang_vel = saved_vel_refs[1::2]
            self.solution_space_violation = np.fmax(0.0, np.fabs(saved_u[0::2]/1.5) + np.fabs(0.5*saved_u[1::2]/1.5) - 1)

        # Position error (squared euclidan)
        dx = self.saved_traj_refs[:, :2] - self.states[:, :2]
        self.error = np.einsum('ij,ij->i', dx, dx)
        
        # #### Create Gui Elements ###########
        self.mainbox = QtGui.QWidget()
        self.setCentralWidget(self.mainbox)
        self.mainbox.setLayout(QtGui.QVBoxLayout())

        self.canvas = pg.GraphicsLayoutWidget()
        self.mainbox.layout().addWidget(self.canvas)

        self.label = QtGui.QLabel()
        self.mainbox.layout().addWidget(self.label)

        # Set up the trajectory plot
        self.trajPlot = self.canvas.addPlot(title='Trajectories')
        xmax = np.amax(self.trajectories[:, 0])
        xmin = np.amin(self.trajectories[:, 0])
        ymax = np.amax(self.trajectories[:, 1])
        ymin = np.amin(self.trajectories[:, 1])
        self.trajPlot.setXRange(xmin, xmax, padding=0.1)
        self.trajPlot.setYRange(ymin, ymax, padding=0.1)
        self.trajPlot.showGrid(x=True, y=True)
        self.trajPlot.setLabel('left', "y-pos", units='m')
        self.trajPlot.setLabel('bottom', "x-pos", units='m')
        if self.plot_details:
        # Set up the posistion error plot
            self.canvas.nextCol()
            self.posErrorPlot = self.canvas.addPlot(title="xy position error")
            self.posErrorPlot.showGrid(x=True, y=True)
            self.posErrorPlot.setLabel('left', "Position error", units='m')
            self.posErrorPlot.setLabel('bottom', "Time", units='s')
            # Set up the solution space plot
            self.canvas.nextCol()
            self.solutionSpacePlot = self.canvas.addPlot(title="Solution space")
            self.solutionSpacePlot.showGrid(x=True, y=True)
            self.solutionSpacePlot.setLabel('left', "Linear velocity", units='m/s')
            self.solutionSpacePlot.setLabel('bottom', "Angular velocity", units='rad/s')
            # Set up the linear velocity plot
            self.canvas.nextRow()
            self.linVelPlot = self.canvas.addPlot(title = "Linear velocity")
            self.linVelPlot.showGrid(x=True, y=True)
            self.linVelPlot.setLabel('left', units='m/s')
            self.linVelPlot.setLabel('bottom', "Time", units='s')
            # Set up the angular velocity plot
            self.canvas.nextCol()
            self.angVelPlot = self.canvas.addPlot(title = "Angular velocity")
            self.angVelPlot.showGrid(x=True, y=True)
            self.angVelPlot.setLabel('left', "Angular velocity", units='rad/s')
            self.angVelPlot.setLabel('bottom', "Time", units='s')
            # Set up solution space violation plot
            self.canvas.nextCol()
            self.solutionSpaceViolationPlot = self.canvas.addPlot(title = "Violation of the solution space constraints")
            self.solutionSpaceViolationPlot.showGrid(x=True, y=True)
            self.solutionSpaceViolationPlot.setLabel('left', "Constraint violation")
            self.solutionSpaceViolationPlot.setLabel('bottom', "Time", units='s')
        
        
        self.traj = []
        color = ['y', 'r', 'g', 'b']
        for i in range(self.N_traj):
            self.traj += [self.trajPlot.plot(pen=color[i%len(color)], symbol='o', symbolSize=5, symbolBrush=color[i%len(color)])]

        self.cur_atr_state = self.trajPlot.plot( symbol='o', symbolBrush ='r', name='Current state')
        self.prev_atr_states = self.trajPlot.plot(name='Traversed path')
        if self.plot_details:
            self.error_line = self.posErrorPlot.plot(pen='y', name='Position error')
            self.cur_error = self.posErrorPlot.plot(symbol='o', symbolBrush ='r', name='Current error')

            self.solution_space = self.solutionSpacePlot.plot(pen='r', name='Solution space limits')
            self.solution_space_line = self.solutionSpacePlot.plot(pen='y', name='Previous control signals')
            self.cur_solution_space = self.solutionSpacePlot.plot(symbol='o', symbolBrush ='r', name='Current control signals')

            self.solution_space_violation_line = self.solutionSpaceViolationPlot.plot(pen='y', name="Constraint violation")
            self.cur_solution_space_violation = self.solutionSpaceViolationPlot.plot(symbol='o', symbolBrush ='r', name="Current violation")

            self.ref_lin_vel_line = self.linVelPlot.plot(pen='b', name='Lin. vel. reference')
            self.lin_vel_line = self.linVelPlot.plot(pen='y', name='Linear velocity')
            self.cur_lin_vel = self.linVelPlot.plot(symbol='o', symbolBrush ='r', name='Current lin. vel.')

            self.ref_ang_vel_line = self.angVelPlot.plot(pen='b', name='Ang. vel. reference')
            self.ang_vel_line = self.angVelPlot.plot(pen='y', name='Angular velocity')        
            self.cur_ang_vel = self.angVelPlot.plot(symbol='o', symbolBrush ='r', name="Current ang. vel.")

            #### Set Data  #####################
            self.t = np.linspace(self.ts, self.sim_time, self.Nsim)
            self.error_line.setData(self.t, self.error)

            self.lin_vel_line.setData(self.t, self.lin_vel)
            self.ref_lin_vel_line.setData(self.t, self.ref_lin_vel)

            self.ang_vel_line.setData(self.t, self.ang_vel)
            self.ref_ang_vel_line.setData(self.t, self.ref_ang_vel)

            self.solution_space.setData([0, 3.0, 0, -3.0, 0], [-0.5, 0, 1.5, 0, -0.5])
            self.solution_space_violation_line.setData(self.t, self.solution_space_violation)

        self.counter = 0
        self.fps = 0.
        self.lastupdate = time.time()
        self.plot_idx = self.active_traj * self.N

        #### Start  #####################
        self.start_time = time.time()
        self._update()

    def _update(self):
        t = self.counter * self.ts
        try: # For dealing with indexing bug...
            self.cur_atr_state.setData([self.states[self.counter, 0]], [self.states[self.counter, 1]])
            self.prev_atr_states.setData(self.states[:self.counter, 0], self.states[:self.counter, 1])
            if self.plot_details:            
                self.cur_error.setData([t], [self.error[self.counter]])
                self.cur_lin_vel.setData([t], [self.lin_vel[self.counter]])
                self.cur_ang_vel.setData([t], [self.ang_vel[self.counter]])
                self.cur_solution_space.setData([self.ang_vel[self.counter]], [self.lin_vel[self.counter]])
                self.solution_space_line.setData(self.ang_vel[:self.counter], self.lin_vel[:self.counter])
                self.cur_solution_space_violation.setData([t], [self.solution_space_violation[self.counter]])     
        
            if self.counter in self.new_traj_idx:
                self.active_traj += 1
                self.plot_idx = self.active_traj * self.traj_len
                i = 0
                for line in self.traj:
                    if i == self.active_traj:
                        line.setSymbolBrush('g')
                        line.setPen('g')
                        line.setSymbolSize(5)
                    else:   
                        line.setSymbolBrush('r')
                        line.setPen('r')
                        line.setSymbolSize(3)
                    i += 1
                self.traj[self.active_traj].setData(self.trajectories[self.plot_idx:self.plot_idx + self.traj_len, :])
                
            now = time.time()
            dt = (now-self.lastupdate)
            if dt <= 0:
                dt = 0.000000000001
            fps2 = 1.0 / dt
            self.lastupdate = now
            self.fps = self.fps * 0.9 + fps2 * 0.1
            tx = 'Mean Frame Rate:  {fps:.3f} FPS'.format(fps=self.fps)
            
            sim_time = self.counter * 0.01
            real_time = time.time() - self.start_time
            speed = sim_time / real_time
            txt = f'Simulation time elapsed: {round(sim_time, 2)}, plots are moving at {round(speed,2)}x real time'
            self.label.setText(txt)
            QtCore.QTimer.singleShot(1, self._update)
            self.counter += 1

            # There is a bug here that sometimes causes the plotter to not reset properly
            # which then causes a crash in the next iteration
            if t + self.ts >= (self.Nsim)* self.ts:
                
                for line in self.traj:
                    line.clear()
                self.active_traj = -1
                self.start_time = time.time()
                self.counter = -1
                self.plot_idx = self.active_traj * self.N
        except: # Ugly way to deal with the bug instead of fixing it..
            self.logger.info("Bug occured, resetting plots via try/except")
            for line in self.traj:
                line.clear()
            self.active_traj = -1
            self.start_time = time.time()
            self.counter = -1
            self.plot_idx = self.active_traj * self.N
            self._update()
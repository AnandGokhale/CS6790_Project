import numpy as np
import g2o
import time
import os

from collections import namedtuple

params = {
    'pnp_min_measurements': 10,
    'pnp_max_iterations': 10,
    'frustum_near': 0.1,
    'frustum_far': 50.0
}

class BundleAdjustment(g2o.SparseOptimizer):
    def __init__(self, ):
        super().__init__()

        # Higher confident (better than CHOLMOD, according to 
        # paper "3-D Mapping With an RGB-D Camera")
        solver = g2o.BlockSolverSE3(g2o.LinearSolverCSparseSE3())
        solver = g2o.OptimizationAlgorithmLevenberg(solver)
        super().set_algorithm(solver)

        # Convergence Criterion
        terminate = g2o.SparseOptimizerTerminateAction()
        terminate.set_gain_threshold(1e-6)
        super().add_post_iteration_action(terminate)

        # Robust cost Function (Huber function) delta
        self.delta = np.sqrt(5.991)   
        self.aborted = False

    def optimize(self, max_iterations=10):
        super().initialize_optimization()
        super().optimize(max_iterations)
        try:
            return not self.aborted
        finally:
            self.aborted = False

    def add_pose(self, pose_id, pose, cam, fixed=False):
        sbacam = g2o.SBACam(
            pose.orientation(), pose.position())
        sbacam.set_cam(
            cam.fx, cam.fy, cam.cx, cam.cy, cam.baseline)

        v_se3 = g2o.VertexCam()
        v_se3.set_id(pose_id * 2)
        v_se3.set_estimate(sbacam)
        v_se3.set_fixed(fixed)
        super().add_vertex(v_se3) 

    def add_point(self, point_id, point, fixed=False, marginalized=True):
        v_p = g2o.VertexSBAPointXYZ()
        v_p.set_id(point_id * 2 + 1)
        v_p.set_marginalized(marginalized)
        v_p.set_estimate(point)
        v_p.set_fixed(fixed)
        super().add_vertex(v_p)

    def add_edge(self, id, point_id, pose_id, meas):
        if meas.is_stereo():
            edge = self.stereo_edge(meas.xyx)
        elif meas.is_left():
            edge = self.mono_edge(meas.xy)
        elif meas.is_right():
            edge = self.mono_edge_right(meas.xy)

        edge.set_id(id)
        edge.set_vertex(0, self.vertex(point_id * 2 + 1))
        edge.set_vertex(1, self.vertex(pose_id * 2))
        kernel = g2o.RobustKernelHuber(self.delta)
        edge.set_robust_kernel(kernel)
        super().add_edge(edge)

    def stereo_edge(self, projection, information=np.identity(3)):
        e = g2o.EdgeProjectP2SC()
        e.set_measurement(projection)
        e.set_information(information)
        return e

    def mono_edge(self, projection, 
            information=np.identity(2) * 0.5):
        e = g2o.EdgeProjectP2MC()
        e.set_measurement(projection)
        e.set_information(information)
        return e

    def mono_edge_right(self, projection, 
            information=np.identity(2) * 0.5):
        e = g2o.EdgeProjectP2MCRight()
        e.set_measurement(projection)
        e.set_information(information)
        return e

    def get_pose(self, id):
        return self.vertex(id * 2).estimate()

    def get_point(self, id):
        return self.vertex(id * 2 + 1).estimate()

    def abort(self):
        self.aborted = True

class MotionModel(object):

    def __init__(self, 
            timestamp=None, 
            initial_position=np.zeros(3), 
            initial_orientation=g2o.Quaternion(), 
            initial_covariance=None):

        self.timestamp = timestamp
        self.position = initial_position
        self.orientation = initial_orientation
        self.covariance = initial_covariance    # pose covariance

        self.v_linear = np.zeros(3)    # linear velocity
        self.v_angular_angle = 0
        self.v_angular_axis = np.array([1, 0, 0])

        self.initialized = False
        # damping factor
        self.damp = 0.95

    def current_pose(self):
        '''
        Get the current camera pose.
        '''
        return (g2o.Isometry3d(self.orientation, self.position), 
            self.covariance)

    def predict_pose(self, timestamp):
        '''
        Predict the next camera pose.
        '''
        if not self.initialized:
            return (g2o.Isometry3d(self.orientation, self.position), 
                self.covariance)
        
        dt = timestamp - self.timestamp

        delta_angle = g2o.AngleAxis(
            self.v_angular_angle * dt * self.damp, 
            self.v_angular_axis)
        delta_orientation = g2o.Quaternion(delta_angle)

        position = self.position + self.v_linear * dt * self.damp
        orientation = self.orientation * delta_orientation

        return (g2o.Isometry3d(orientation, position), self.covariance)

    def update_pose(self, timestamp, 
            new_position, new_orientation, new_covariance=None):
        '''
        Update the motion model when given a new camera pose.
        '''
        if self.initialized:
            dt = timestamp - self.timestamp
            assert dt != 0

            v_linear = (new_position - self.position) / dt
            self.v_linear = v_linear

            delta_q = self.orientation.inverse() * new_orientation
            delta_q.normalize()

            delta_angle = g2o.AngleAxis(delta_q)
            angle = delta_angle.angle()
            axis = delta_angle.axis()

            if angle > np.pi:
                axis = axis * -1
                angle = 2 * np.pi - angle

            self.v_angular_axis = axis
            self.v_angular_angle = angle / dt
            
        self.timestamp = timestamp
        self.position = new_position
        self.orientation = new_orientation
        self.covariance = new_covariance
        self.initialized = True

    def apply_correction(self, correction):     # corr: g2o.Isometry3d or matrix44
        '''
        Reset the model given a new camera pose.
        Note: This method will be called when it happens an abrupt change in the pose (LoopClosing)
        '''
        if not isinstance(correction, g2o.Isometry3d):
            correction = g2o.Isometry3d(correction)

        current = g2o.Isometry3d(self.orientation, self.position)
        current = current * correction

        self.position = current.position()
        self.orientation = current.orientation()

        self.v_linear = (
            correction.inverse().orientation() * self.v_linear)
        self.v_angular_axis = (
            correction.inverse().orientation() * self.v_angular_axis)

class Camera(object):
    def __init__(self, fx, fy, cx, cy, width, height, 
            frustum_near, frustum_far, baseline):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.baseline = baseline

        self.intrinsic = np.array([
            [fx, 0, cx], 
            [0, fy, cy], 
            [0, 0, 1]])

        self.frustum_near = frustum_near
        self.frustum_far = frustum_far

        self.width = width
        self.height = height
        
    def compute_right_camera_pose(self, pose):
        pos = pose * np.array([self.baseline, 0, 0])
        return g2o.Isometry3d(pose.orientation(), pos)

class Tracking(object):
    def __init__(self, params):
        self.optimizer = BundleAdjustment()
        self.min_measurements = params['pnp_min_measurements']
        self.max_iterations = params['pnp_max_iterations']

    def refine_pose(self, pose, cam, measurements):
        assert len(measurements) >= self.min_measurements, (
            'Not enough points')
            
        self.optimizer.clear()
        self.optimizer.add_pose(0, pose, cam, fixed=False)

        for i, m in enumerate(measurements):
            self.optimizer.add_point(i, m.mappoint.position, fixed=True)
            self.optimizer.add_edge(0, i, 0, m)

        self.optimizer.optimize(self.max_iterations)
        return self.optimizer.get_pose(0)

def cameraParams(path):
    '''
    path example: 'path/to/your/KITTI odometry dataset/sequences/00'
    '''
    Cam = namedtuple('cam', 'fx fy cx cy width height baseline')
    cam00_02 = Cam(718.856, 718.856, 607.1928, 185.2157, 1241, 376, 0.5371657)
    cam03 = Cam(721.5377, 721.5377, 609.5593, 172.854, 1241, 376, 0.53715)
    cam04_12 = Cam(707.0912, 707.0912, 601.8873, 183.1104, 1241, 376, 0.53715)

    path = os.path.expanduser(path)
    sequence = int(path.strip(os.path.sep).split(os.path.sep)[-1])

    if sequence < 3:
        return cam00_02
    elif sequence == 3:
        return cam03
    elif sequence < 13:
        return cam04_12


path = 'path/to/your/KITTI odometry dataset/sequences/00'
camera = cameraParams(path)
cam = Camera(
    camera.fx, 
    camera.fy, 
    camera.cx, 
    camera.cy, 
    camera.width, 
    camera.height, 
    params['frustum_near'], 
    params['frustum_far'], 
    camera.baseline
    )
tracker = Tracking(params)

# measurements have to be the way sptam repo requires.
def estimatePose(tracker=tracker, cur_pose=g2o.Isometry3d(), cam=cam, measurements=[]):
    new_pose = tracker.refine_pose(cur_pose, cam, measurements)
    return new_pose.orientation(), new_pose.position()
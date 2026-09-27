import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from fingr.constants import ARM_JOINT_NAMES, SHARPA_RIGHT_JOINT_NAMES


DEFAULT_ARM_URDF = Path(__file__).resolve().parents[1] / "assets/xarm7_sharpa/xarm7_sharpa_right.urdf"


def rpy_to_matrix(rpy):
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def rotvec_to_matrix(rv):
    rv = np.asarray(rv, np.float64)
    single = rv.ndim == 1
    rv = rv.reshape(-1, 3)
    theta = np.linalg.norm(rv, axis=1, keepdims=True)
    axis = rv / np.maximum(theta, 1e-12)
    k = np.zeros((len(rv), 3, 3))
    k[:, 0, 1], k[:, 0, 2] = -axis[:, 2], axis[:, 1]
    k[:, 1, 0], k[:, 1, 2] = axis[:, 2], -axis[:, 0]
    k[:, 2, 0], k[:, 2, 1] = -axis[:, 1], axis[:, 0]
    s = np.sin(theta)[..., None]
    c = np.cos(theta)[..., None]
    r = np.eye(3) + s * k + (1 - c) * (k @ k)
    return r[0] if single else r


def transform(rotation, translation):
    t = np.eye(4)
    t[:3, :3] = rotation
    t[:3, 3] = translation
    return t


ARM_TO_WORLD = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], np.float64)
T_ARM_WORLD = transform(ARM_TO_WORLD, np.zeros(3))


class ArmHandModel:
    def __init__(self, urdf=DEFAULT_ARM_URDF):
        root = ET.parse(urdf).getroot()
        joints = root.findall("joint")
        self.children = {}
        self.origins = {}
        self.axes = {}
        for joint in joints:
            origin = joint.find("origin")
            axis = joint.find("axis")
            xyz = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ") if origin is not None else np.zeros(3)
            rpy = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ") if origin is not None else np.zeros(3)
            name = joint.get("name")
            parent = joint.find("parent").get("link")
            child = joint.find("child").get("link")
            self.origins[name] = transform(rpy_to_matrix(rpy), xyz)
            self.axes[name] = np.fromstring(axis.get("xyz"), sep=" ") if axis is not None else np.array([0, 0, 1.0])
            self.children.setdefault(parent, []).append((name, joint.get("type"), child))
        self.t_link7_hand = self.origins["joint_eef"]

    def link_transforms(self, arm_qpos, hand_qpos, t_world_arm=T_ARM_WORLD, t_link7_hand=None):
        arm_qpos = np.asarray(arm_qpos, np.float64)
        hand_qpos = np.asarray(hand_qpos, np.float64)
        single = arm_qpos.ndim == 1
        arm_qpos = np.atleast_2d(arm_qpos)
        hand_qpos = np.atleast_2d(hand_qpos)
        n = len(arm_qpos)
        t_link7_hand = self.t_link7_hand if t_link7_hand is None else t_link7_hand
        angles = {name: arm_qpos[:, i] for i, name in enumerate(ARM_JOINT_NAMES)}
        angles.update({name: hand_qpos[:, i] for i, name in enumerate(SHARPA_RIGHT_JOINT_NAMES)})
        poses = {"link_base": np.tile(t_world_arm, (n, 1, 1))}
        stack = ["link_base"]
        while stack:
            parent = stack.pop()
            for name, joint_type, child in self.children.get(parent, []):
                pose = np.tile(t_link7_hand if name == "joint_eef" else self.origins[name], (n, 1, 1))
                if joint_type in ("revolute", "continuous"):
                    pose[:, :3, :3] = pose[:, :3, :3] @ rotvec_to_matrix(angles[name][:, None] * self.axes[name])
                poses[child] = poses[parent] @ pose
                stack.append(child)
        return {name: pose[0] if single else pose for name, pose in poses.items()}

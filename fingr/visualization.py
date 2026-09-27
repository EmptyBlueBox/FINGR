from pathlib import Path

import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from fingr.constants import ARM_JOINT_NAMES, SHARPA_RIGHT_JOINT_NAMES, TACTILE_NAMES
from fingr.kinematics import DEFAULT_ARM_URDF, T_ARM_WORLD, rotvec_to_matrix, rpy_to_matrix, transform


ARMHAND_URDF = Path(DEFAULT_ARM_URDF)
ROBOT_JOINT_NAMES = [*ARM_JOINT_NAMES, *SHARPA_RIGHT_JOINT_NAMES]


def armhand_overrides(path="world/armhand"):
    root = f"{path}/{ARMHAND_URDF.stem}"
    return {
        f"{root}/{geometry}_geometries": rrb.EntityBehavior(visible=geometry == "visual")
        for geometry in ["collision", "visual"]
    }


def log_armhand_joints(joints, qpos, path="world/armhand"):
    values = dict(zip(ROBOT_JOINT_NAMES, qpos))
    for name, joint in joints.items():
        if joint.joint_type in ("revolute", "continuous"):
            rr.log(path, joint.compute_transform(float(values.get(name, 0.0))))


def init_armhand(qpos, path="world/armhand", static=True):
    rr.log_file_from_path(ARMHAND_URDF, entity_path_prefix=path, static=True)
    tree = rr.urdf.UrdfTree.from_file_path(ARMHAND_URDF)
    rr.log(path, rr.Transform3D.from_fields(translation=T_ARM_WORLD[:3, 3], mat3x3=T_ARM_WORLD[:3, :3], parent_frame=f"tf#{path}", child_frame=tree.root_link().name), static=static)
    joints = {joint.name: joint for joint in tree.joints()}
    log_armhand_joints(joints, qpos, path)
    return joints


TACTILE_TREE = rr.urdf.UrdfTree.from_file_path(DEFAULT_ARM_URDF)
TACTILE_JOINTS = TACTILE_TREE.joints()


def tactile_poses(qpos):
    values = dict(zip(ROBOT_JOINT_NAMES, qpos))
    poses = {TACTILE_TREE.root_link().name: T_ARM_WORLD}
    for _ in TACTILE_JOINTS:
        for joint in TACTILE_JOINTS:
            if joint.parent_link not in poses or joint.child_link in poses:
                continue
            pose = transform(rpy_to_matrix(joint.origin_rpy), joint.origin_xyz)
            if joint.joint_type in ("revolute", "continuous"):
                pose = pose @ transform(rotvec_to_matrix(np.asarray(joint.axis) * values.get(joint.name, 0.0)), np.zeros(3))
            poses[joint.child_link] = poses[joint.parent_link] @ pose
    tips = np.stack([poses[f"right_{name}_fingertip"] for name in TACTILE_NAMES])
    elastomers = np.stack([poses[f"right_{name}_elastomer"] for name in TACTILE_NAMES])
    return tips[:, :3, 3], elastomers[:, :3, :3]


def log_tactile(qpos, tactile):
    positions, rotations = tactile_poses(qpos)
    force = -np.einsum("nij,nj->ni", rotations, tactile["tactile_force"])
    magnitude = tactile["tactile_force_magnitude"]
    colors = [[int(255 * min(1, value / 5)), 0, 255 - int(255 * min(1, value / 5))] for value in magnitude]
    rr.log("world/armhand/tactile", rr.Arrows3D(origins=positions, vectors=0.01 * force, radii=np.full(5, 0.002), colors=colors, labels=TACTILE_NAMES, show_labels=False))
    for i, name in enumerate(TACTILE_NAMES):
        rr.log(f"tactile/force_magnitude/{name}", rr.Scalars(float(magnitude[i])))
        heatmap = cv2.applyColorMap(tactile["tactile_heatmap"][i], cv2.COLORMAP_TURBO)
        rr.log(f"tactile/heatmap/{name}", rr.EncodedImage(contents=cv2.imencode(".png", heatmap)[1], media_type="image/png"))

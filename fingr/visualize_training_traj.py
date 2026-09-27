import argparse
from itertools import groupby

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from fingr.constants import TACTILE_NAMES
from fingr.data import ensure_data, open_dataset, sessions
from fingr.kinematics import ArmHandModel
from fingr.policy import cube_angle_to_solved, cube_turn_start, layer_combinations, layer_complements
from fingr.visualization import ROBOT_JOINT_NAMES, armhand_overrides, init_armhand, log_armhand_joints, log_tactile


def cube_mesh(cubies):
    layers, complements = cubies[layer_combinations], cubies[layer_complements]
    _, layer_values, layer_axes = np.linalg.svd(layers - layers.mean(1, keepdims=True), full_matrices=False)
    _, complement_values, complement_axes = np.linalg.svd(complements - complements.mean(1, keepdims=True), full_matrices=False)
    scale = (layer_values[:, 0] + complement_values[:, 0]) / 2
    scores = layer_values[:, 2] + complement_values[:, 2] + abs(layer_values[:, 0] - layer_values[:, 1]) + abs(complement_values[:, 0] - complement_values[:, 1]) + scale * (1 - abs((layer_axes[:, 2] * complement_axes[:, 2]).sum(1)))
    layer = scores.argmin()
    stickers = []
    for points, other in ((layers[layer], complements[layer]), (complements[layer], layers[layer])):
        neighbors = np.linalg.norm(points[:, None] - points[None], axis=2).argsort(1)[:, 1:3]
        outward = (points.mean(0) - other.mean(0)) / 2
        for point, adjacent in zip(points, points[neighbors]):
            axes = np.vstack([(point - adjacent) / 2, outward])
            for i in range(3):
                normal, a, b = axes[i], axes[(i + 1) % 3], axes[(i + 2) % 3]
                if np.cross(a, b) @ normal < 0:
                    a, b = b, a
                stickers.append(point + normal + np.array([-a - b, -a + b, a + b, a - b]))
    return mesh(np.asarray(stickers), np.eye(3), np.zeros(3))


def mesh(stickers, rotation, translation):
    vertices = (stickers @ rotation.T + translation).reshape(-1, 3).astype(np.float32)
    triangles = (np.arange(len(stickers))[:, None, None] * 4 + np.array([[0, 1, 2], [0, 2, 3]])).reshape(-1, 3)
    normals = np.cross(stickers[:, 2] - stickers[:, 0], stickers[:, 1] - stickers[:, 0])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    return vertices, triangles, np.full(vertices.shape, 150, np.uint8), np.repeat(normals, 4, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/training')
    parser.add_argument('--episode', type=int, default=0)
    parser.add_argument('--no-spawn', action='store_true')
    parser.add_argument('--rrd', default='')
    args = parser.parse_args()
    path = ensure_data(args.data)
    dataset = open_dataset(path, [args.episode])
    session = sessions(path)[args.episode]
    rr.init('FINGR')
    if not args.no_spawn:
        rr.spawn(connect=True, memory_limit='2GiB', server_memory_limit='512MiB', hide_welcome_screen=True)
    if args.rrd:
        rr.save(args.rrd)
    rr.log('', rr.ViewCoordinates.RUF, static=True)
    rr.log('world', rr.ViewCoordinates.RUF, static=True)
    scene = rrb.Spatial3DView(origin='/world', name='armhand and cube', overrides=armhand_overrides())
    heatmaps = rrb.Grid(contents=[rrb.Spatial2DView(origin=f'/tactile/heatmap/{name}', name=name) for name in TACTILE_NAMES], grid_columns=5)
    signals = rrb.Tabs(*[rrb.TimeSeriesView(origin=f'/training/{name}', name=name) for name in ('angle_to_solved', 'qvel', 'torque', 'action', 'tactile_force')])
    signals = rrb.Vertical(signals, rrb.StateTimelineView(origin='/training/steps', name='turns'), row_shares=[0.8, 0.2])
    rr.send_blueprint(rrb.Blueprint(rrb.Horizontal(rrb.Vertical(scene, heatmaps, rrb.TextDocumentView(origin='/training/status'), row_shares=[0.56, 0.28, 0.16]), signals, column_shares=[0.65, 0.35]), collapse_panels=True))
    model = ArmHandModel()
    step_ends = [step['end_frame'] for step in session['steps']]
    formula = ''.join(move + (str(count) if count > 1 else '') for move, count in ((move, len(list(group))) for move, group in groupby(session['moves'])))
    current_step = -1
    turn_start = None
    for item in dataset:
        frame = int(item['frame_index'])
        sample_time = int(np.asarray(item['observation.sample_time_ns']).item())
        qpos = np.asarray(item['observation.state'])
        cubies = np.asarray(item['observation.cube_cubie_positions'])
        cube_valid = bool(item['observation.cube_valid'])
        force = np.asarray(item['observation.tactile_force'])
        heatmap = np.stack([item[f'observation.images.tactile_heatmap_{name}'][0].numpy() for name in TACTILE_NAMES])
        step_index = int(np.searchsorted(step_ends, frame, side='right'))
        step = session['steps'][step_index]
        move = step['move']
        rr.set_time('frame', sequence=frame)
        if frame == 0:
            first_sample = sample_time
            joints = init_armhand(qpos, static=False)
        rr.set_time('time', duration=(sample_time - first_sample) / 1e9)
        if step_index != current_step:
            turn_start = None
            previous_rotation = 0
            current_step = step_index
            rr.log('training/steps/formula', rr.StateChange(state=f'{move} {step_index + 1}/{len(step_ends)}'))
        if not cube_valid or not np.isfinite(cubies).all():
            angle = np.nan
        elif 'angle_to_solved_deg' in session:
            value = session['angle_to_solved_deg'][frame]
            angle = float(value) if value is not None else np.nan
        else:
            if turn_start is None:
                palm = model.link_transforms(qpos[:7], qpos[7:])['right_hand_C_MC']
                turn_start = cube_turn_start((cubies - palm[:3, 3]) @ palm[:3, :3], move)
            angle, previous_rotation = cube_angle_to_solved(cubies, turn_start, move, previous_rotation)
            angle = np.rad2deg(angle)
        log_armhand_joints(joints, qpos)
        log_tactile(qpos, dict(tactile_force=force[:, :3], tactile_force_magnitude=np.linalg.norm(force[:, :3], axis=1), tactile_heatmap=heatmap))
        if np.isfinite(cubies).all():
            vertices, triangles, colors, normals = cube_mesh(cubies)
            rr.log('world/cube', rr.Mesh3D(vertex_positions=vertices, triangle_indices=triangles, vertex_normals=normals, vertex_colors=colors))
        else:
            rr.log('world/cube', rr.Clear(recursive=True))
        status = f"episode: {args.episode}  frame: {frame}/{len(dataset)}  formula: {formula}\nturn: {step_index + 1}/{len(step_ends)} {move}  cube valid: {cube_valid}\nremaining angle: {angle:.1f} deg  completed: {step['completed']}"
        rr.log('training/status', rr.TextDocument(status))
        rr.log('training/angle_to_solved', rr.Scalars(float(angle)))
        for field, key in (('qvel', 'observation.qvel'), ('torque', 'observation.torque'), ('action', 'action')):
            for name, value in zip(ROBOT_JOINT_NAMES, np.asarray(item[key])):
                rr.log(f'training/{field}/{name}', rr.Scalars(float(value)))
        for name, values in zip(TACTILE_NAMES, force):
            for axis, value in zip(('fx', 'fy', 'fz', 'tx', 'ty', 'tz'), values):
                rr.log(f'training/tactile_force/{name}/{axis}', rr.Scalars(float(value)))
    rr.set_time('frame', sequence=frame + 1)
    rr.set_time('time', duration=(sample_time - first_sample) / 1e9 + 1 / dataset.fps)
    rr.log('training/steps/formula', rr.StateChange(state=''))
    print(f'episode={args.episode} frames={len(dataset)}', flush=True)


if __name__ == '__main__':
    main()

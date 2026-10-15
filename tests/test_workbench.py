"""Physics, transaction, persistence and recording acceptance tests on the actual MuJoCo backend."""
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

import mujoco
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
from studio_world import World, STATE, WORLD_XML
from studio_project import read_archive, save_archive


class WorkbenchTests(unittest.TestCase):
    def setUp(self):
        self.world = World()
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)

    def tearDown(self):
        self.world.close()
        self.directory.cleanup()

    def test_free_fall_matches_reference_and_settles_on_floor(self):
        w = self.world
        reference = mujoco.MjData(w.model)
        for _ in range(500):
            w.step()
            mujoco.mj_step(w.model, reference)
        np.testing.assert_allclose(w.data.qpos, reference.qpos, atol=1e-12)
        self.assertAlmostEqual(w.data.qpos[2], .2, delta=.01)
        self.assertGreater(w.data.ncon, 0)
        self.assertGreater(w.status()['contact_force'], 0)

    def test_edit_undo_redo_restores_mass_position_and_state(self):
        w = self.world
        original = w.capture_state()
        w.body_edit({'id': 1, 'mass': 3, 'position': [1, 2, 3], 'friction': .2})
        self.assertEqual(w.model.body_mass[1], 3)
        np.testing.assert_allclose(w.data.qpos[:3], [1, 2, 3])
        w.undo()
        self.assertEqual(w.model.body_mass[1], 1)
        np.testing.assert_array_equal(w.capture_state(), original)
        w.undo(redo=True)
        self.assertEqual(w.model.body_mass[1], 3)
        np.testing.assert_allclose(w.data.qpos[:3], [1, 2, 3])

    def test_failed_edit_keeps_original_world_and_history(self):
        w = self.world
        original = w.capture_state()
        revision = w.revision
        def invalid(spec):
            spec.worldbody.add_body(name='invalid').add_freejoint()
        with self.assertRaises(ValueError): w.edit(invalid)
        self.assertEqual(w.revision, revision)
        np.testing.assert_array_equal(w.capture_state(), original)
        self.assertFalse(w.undo_stack)

    def test_real_geometry_and_undo(self):
        w = self.world
        w.add({'shape': 'sphere', 'position': [1, 0, 1]})
        self.assertEqual(w.model.nbody, 3)
        w.step(500)
        self.assertAlmostEqual(w.data.qpos[-5], .15, delta=.02)
        w.undo()
        self.assertEqual(w.model.nbody, 2)

    def test_sensor_outputs_and_force_duration(self):
        w = self.world
        w.add_imu({'id': 1})
        self.assertEqual(w.model.nsensor, 2)
        w.force({'id': 1, 'force': [10, 0, 0], 'duration': .1})
        w.step(100)
        self.assertGreater(w.data.qvel[0], .9)
        np.testing.assert_array_equal(w.data.xfrc_applied, 0)
        self.assertEqual(len(w.status()['sensordata']), 6)

    def test_project_roundtrip_at_contact_restores_state_and_camera(self):
        w = self.world
        w.step(500)
        w.camera.azimuth = 73
        state = w.capture_state()
        path = self.root / 'contact.mjstudio'
        w.save(str(path))
        w.step(10)
        w.open(str(path))
        np.testing.assert_array_equal(w.capture_state(), state)
        self.assertEqual(w.camera.azimuth, 73)
        self.assertFalse(w.dirty)

    def test_record_replay_csv_does_not_mutate_live_world(self):
        w = self.world
        w.start_recording()
        w.step(500)
        w.recording = False
        self.assertEqual(len(w.record_times), 51)
        state = w.capture_state()
        w.export_csv(str(self.root/'data.csv'))
        np.testing.assert_array_equal(w.capture_state(), state)
        with (self.root/'data.csv').open() as file:
            rows = list(csv.reader(file))
        self.assertEqual(len(rows), len(w.record_times)+1)
        self.assertEqual(rows[0][0], 'time_s')
        path = self.root/'run.mjrec'
        w.export_recording(str(path))
        stored = np.asarray(w.record_states).copy()
        w.open(str(path), recording=True)
        w.seek(12)
        np.testing.assert_array_equal(w.capture_state(), stored[12])
        self.assertTrue(w.status()['replaying'])

    def test_invalid_project_does_not_replace_world(self):
        w = self.world
        path = self.root/'bad.mjstudio'
        save_archive(str(path), w.spec, {'kind': 'project', 'state': [1,2,3]})
        original = w.capture_state()
        with self.assertRaises(ValueError): w.open(str(path))
        np.testing.assert_array_equal(w.capture_state(), original)

    def test_invalid_settings_and_running_edit_rejected(self):
        w = self.world
        for timestep in (0, -.1, float('nan'), float('inf')):
            with self.assertRaises(ValueError): w.physics({'timestep': timestep})
        w.running = True
        with self.assertRaises(ValueError): w.add({'shape': 'box'})
        w.running = False
        w.start_recording()
        with self.assertRaises(ValueError): w.body_edit({'id': 1, 'mass': 2})

    def test_joint_motor_control_clips_input_and_pd_tracks(self):
        w = self.world
        w.load(builtin='cartpole')
        w.set_joint({'id': 0, 'value': .4})
        w.control({'id': 0, 'mode': 'direct', 'target': 500})
        w.step()
        self.assertEqual(w.data.ctrl[0], 1)
        w.reset()
        w.control({'id': 0, 'mode': 'pd', 'target': .2, 'kp': 20, 'kd': 10})
        w.step(20)
        self.assertGreater(w.data.qpos[0], 0)

    def test_position_servo_rejects_motor_pd(self):
        w = self.world
        w.install(mujoco.MjSpec.from_string('<mujoco><worldbody><body><joint name="j"/><geom type="sphere" size=".1"/></body></worldbody><actuator><position name="p" joint="j" kp="10"/></actuator></mujoco>'))
        self.assertFalse(w.pd_compatible(0))
        with self.assertRaises(ValueError): w.control({'id': 0, 'mode': 'pd', 'target': 1})
        w.control({'id': 0, 'mode': 'direct', 'target': .3})
        w.step(50)
        self.assertGreater(w.data.qpos[0], 0)

    def test_g1_archive_embeds_assets_and_loads_without_original_paths(self):
        w = self.world
        w.load(builtin='g1')
        path = self.root/'robot.mjstudio'
        w.save(str(path))
        with zipfile.ZipFile(path) as archive:
            xml = archive.read('scene.xml').decode()
            self.assertNotIn(str(Path.cwd()), xml)
            self.assertGreater(len([n for n in archive.namelist() if n.startswith('assets/')]), 20)
        w.load()
        w.open(str(path))
        self.assertEqual(w.model.nu, 29)
        self.assertEqual(Image.open(io.BytesIO(w.render())).size, (960,640))

    def test_recording_limit_stops_without_dropping_recorded_data(self):
        w = self.world
        w.start_recording()
        w.record_limit = 3
        w.step(100)
        self.assertFalse(w.recording)
        self.assertEqual(len(w.record_times), 3)
        self.assertTrue(w.last_notice)

    def test_unnamed_joints_get_stable_names(self):
        w = self.world
        w.install(mujoco.MjSpec.from_string('<mujoco><worldbody><body><joint/><geom type="sphere" size=".1"/></body></worldbody></mujoco>'))
        name = w.model.joint(0).name
        self.assertTrue(name)
        w.set_joint({'id':0, 'value':.5})
        w.add({'shape':'sphere'})
        self.assertAlmostEqual(w.data.qpos[0], .5)

    def test_multiple_robots_keep_offset_pose_and_portable_assets(self):
        w = self.world
        w.insert_model({'builtin':'g1', 'position':[1,0,0]})
        first = w.selected
        j = w.model.body_jntadr[first]
        self.assertAlmostEqual(w.data.qpos[w.model.jnt_qposadr[j]], 1)
        w.reset(key=0)
        self.assertAlmostEqual(w.data.qpos[w.model.jnt_qposadr[j]], 1)
        w.insert_model({'builtin':'g1', 'position':[-1,0,0]})
        self.assertEqual(w.model.nu,58)
        second = w.selected
        j = w.model.body_jntadr[second]
        self.assertAlmostEqual(w.data.qpos[w.model.jnt_qposadr[j]], -1)
        path = self.root/'multi.mjstudio'
        w.save(str(path))
        w.open(str(path))
        self.assertEqual(w.model.nu,58)
        self.assertGreater(len(w.render()),1000)

    def test_trajectory_interpolation_persistence_and_limits(self):
        w = self.world
        w.load(builtin='cartpole')
        path = self.root/'control.csv'
        path.write_text('time_s,ctrl_slide\n0,0\n1,1\n2,0\n')
        w.load_trajectory(str(path))
        w.data.time = .5
        w.apply_controls()
        self.assertAlmostEqual(w.data.ctrl[0], .5)
        project = self.root/'trajectory.mjstudio'
        w.save(str(project));w.open(str(project))
        self.assertEqual(w.trajectory['name'],'control.csv')
        with self.assertRaises(ValueError): w.control({'id':0,'target':0})
        w.data.time = 10
        w.apply_controls()
        self.assertEqual(w.data.ctrl[0],0)
        path.write_text('time_s,ctrl_slide\n1,0\n0,1\n')
        with self.assertRaises(ValueError): w.load_trajectory(str(path))

    def test_quaternion_edit_and_undo(self):
        w = self.world
        w.body_edit({'id':1, 'quaternion':[2,0,0,2]})
        np.testing.assert_allclose(w.data.qpos[3:7], [2**-.5,0,0,2**-.5])
        w.undo()
        np.testing.assert_array_equal(w.data.qpos[3:7],[1,0,0,0])
        with self.assertRaises(ValueError): w.body_edit({'id':1,'quaternion':[0,0,0,0]})


if __name__ == '__main__': unittest.main(verbosity=2)

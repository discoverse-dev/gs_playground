import argparse
import numpy as np
from gs_playground.src.nav_demo.utils.controller import KeyboardCommandAdapter
from gs_playground.src.nav_demo.utils.policy import G1LocomotionPolicy, Go1LocomotionPolicy, Go2LocomotionPolicy
from gs_playground.src.nav_demo.utils.robot import G1Robot, Go1Robot, Go2Robot
from motrixsim import SceneData, msd, run
from motrixsim.render import RenderApp, Layout

camera_positions = {"g1": [-1.5, 0, 1.0], "go1": [-2, 0, 0.5], "go2": [-2, 0, 0.5]}


def main():
    parser = argparse.ArgumentParser(description="Keyboard control for robots")
    parser.add_argument("--robot", type=str, choices=["g1", "go1", "go2"], default="go2")
    parser.add_argument("--scene", type=str, default="plane", help="Path to scene XML file")
    parser.add_argument("--gs_ply", type=str, default="", help="Path to background gaussian splatting ply file")
    parser.add_argument("--no-sync", action="store_true", help="Disable real-time clock sync")
    args = parser.parse_args()

    scene_file = "gs_playground/models/robots/navigation/flat_scene.xml" if args.scene == "plane" else args.scene

    if args.robot == "g1":
        RobotClass = G1Robot
        PolicyClass = G1LocomotionPolicy
    elif args.robot == "go1":
        RobotClass = Go1Robot
        PolicyClass = Go1LocomotionPolicy
    else:
        RobotClass = Go2Robot
        PolicyClass = Go2LocomotionPolicy

    scene = msd.from_file(scene_file)
    robot = msd.from_file(RobotClass.mjcf_path)
    pos = camera_positions[args.robot]
    camera_mjcf = f"""<mujoco model="camera">
  <worldbody>
    <camera name="follower" pos="{" ".join(str(x) for x in pos)}"
      xyaxes="0 -1 0 0 0 1" trackposspeed="2" trackrotspeed="2" />
  </worldbody>
</mujoco>"""

    camera = msd.from_str(camera_mjcf)
    robot.attach(camera, RobotClass.base_link_name)
    scene.attach(robot)
    model = scene.build()

    camera = model.cameras["follower"]
    camera.rotation_track = "look_at_link"
    camera.position_track = "fixed_local"
    camera.track_target_link = model.get_link(RobotClass.base_link_name)

    body = model.get_body(RobotClass.base_link_name)
    robot = RobotClass(body)
    policy = PolicyClass(robot=robot)
    keyboard_adapter = KeyboardCommandAdapter()

    # Initialize Gaussian Renderer if ply file is provided
    gs_renderer = None
    if args.gs_ply:
        from gaussian_renderer import GSRendererMotrixSim
        # We assume the user wants to render the background gs ply file
        gaussians = {"background": args.gs_ply}
        gs_renderer = GSRendererMotrixSim(gaussians, model)

    print(f"Controlling {args.robot.upper()} robot")
    print("=" * 50)
    print("Keyboard Controls:")
    print("  W / Up Arrow    : Forward")
    print("  S / Down Arrow  : Backward")
    print("  Left Arrow      : Strafe Left")
    print("  Right Arrow     : Strafe Right")
    print("  A / D           : Rotate Left / Right")
    print("  ESC             : Exit")
    print("=" * 50)

    with RenderApp() as render:
        render.launch(model)

        data = SceneData(model)
        
        # Get head_camera id if it exists, else assume it's camera index 0 or handle appropriately
        head_camera_id = None
        if "head_camera" in model.cameras:
            head_camera_id = model.cameras["head_camera"].id
        else:
            for i, c in enumerate(model.cameras):
                if c.name.endswith("head_camera"):
                    head_camera_id = i
                    break
        
        # If no head_camera found, try to use a fallback or keep it None
        if head_camera_id is None and len(model.cameras) > 0:
            # Fallback to the first camera that isn't the follower, if available, or just the follower
            head_camera_id = 0

        head_camera_img = np.full((360, 480, 3), [255, 0, 0], dtype=np.uint8)
        head_img = render.create_image(head_camera_img)
        head_widget = render.widgets.create_image_widget(head_img, layout=Layout(left=10, top=10, width=head_camera_img.shape[1], height=head_camera_img.shape[0]))

        bottom_cam_img = np.full((360, 480, 3), [0, 255, 0], dtype=np.uint8)
        bottom_img = render.create_image(bottom_cam_img)
        bottom_widget = render.widgets.create_image_widget(bottom_img, layout=Layout(left=10, top=10+head_camera_img.shape[0], width=bottom_cam_img.shape[1], height=bottom_cam_img.shape[0]))

        system_camera = render.system_camera

        step = [0]
        contrl_dt = 0.02
        n_ctrl = max(1, round(contrl_dt / model.options.timestep))

        def phys_step():
            from motrixsim import step as mstep
            mstep(model, data)
            step[0] += 1
            if step[0] % n_ctrl == 0:
                need_reset = policy.step(data, keyboard_adapter.command)
                if need_reset:
                    data.reset()
                    step[0] = 0

        def render_step():
            keyboard_adapter.update_from_input(render.input)
            if render.input.is_key_just_pressed("escape"):
                return False
            
            # Update GS rendering
            if gs_renderer and head_camera_id is not None:
                gs_renderer.update_gaussians(data)
                # render requires list of camera ids, and resolution H, W
                results = gs_renderer.render(model, data, [head_camera_id, -1], 480, 360, system_camera=system_camera)
                
                if head_camera_id in results:
                    rgb_tensor, _ = results[head_camera_id]
                    # Expected to be (H, W, 3) numpy array
                    rgb_np = rgb_tensor.cpu().numpy()
                    
                    # Ensure range is correct (e.g., 0-255 if directly rendered, or scaled)
                    if rgb_np.dtype != np.uint8:
                        rgb_np = np.clip(rgb_np * 255, 0, 255).astype(np.uint8)
                    
                    # Update widget image
                    head_img.pixels = rgb_np

                if -1 in results:
                    rgb_tensor, _ = results[-1]
                    rgb_np = rgb_tensor.cpu().numpy()
                    
                    if rgb_np.dtype != np.uint8:
                        rgb_np = np.clip(rgb_np * 255, 0, 255).astype(np.uint8)
                    
                    bottom_img.pixels = rgb_np
            
            render.sync(data)
            return True

        run.render_loop(model.options.timestep, 60, phys_step, render_step)


if __name__ == "__main__":
    main()

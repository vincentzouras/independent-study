import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import os

# For callback functions
button_left = False
button_middle = False
button_right = False
lastx = 0
lasty = 0

key_space = False
key_shift = False
key_w = False
key_s = False
key_a = False
key_d = False

MAX_THRUST = 1


def init_controller(model, data):
    """
    Initializes the controller
    """
    global motors_servo_id, motor_right_thrust_id, motor_left_thrust_id
    motors_servo_id = model.actuator("motors_servo").id
    motor_right_thrust_id = model.actuator("motor_right_thrust").id
    motor_left_thrust_id = model.actuator("motor_left_thrust").id


def controller(model, data):
    """
    Sets actuator controls based on keyboard state.
    """
    data.ctrl[:] = 0  # reset controls at start of each step

    if key_space:
        data.ctrl[motors_servo_id] = 0.0
        data.ctrl[motor_right_thrust_id] = MAX_THRUST
        data.ctrl[motor_left_thrust_id] = MAX_THRUST
    elif key_shift:
        current_angle = data.joint("motors_axle").qpos[0]
        target_angle = (
            np.pi if abs(current_angle - np.pi) < abs(current_angle + np.pi) else -np.pi
        )
        data.ctrl[motors_servo_id] = target_angle
        data.ctrl[motor_right_thrust_id] = MAX_THRUST
        data.ctrl[motor_left_thrust_id] = MAX_THRUST
    elif key_w:
        data.ctrl[motors_servo_id] = np.pi / 2
        data.ctrl[motor_right_thrust_id] = MAX_THRUST
        data.ctrl[motor_left_thrust_id] = MAX_THRUST
    elif key_s:
        data.ctrl[motors_servo_id] = -np.pi / 2
        data.ctrl[motor_right_thrust_id] = MAX_THRUST
        data.ctrl[motor_left_thrust_id] = MAX_THRUST
    elif key_a:
        data.ctrl[motors_servo_id] = np.pi / 2
        data.ctrl[motor_right_thrust_id] = MAX_THRUST
        data.ctrl[motor_left_thrust_id] = 0.0
    elif key_d:
        data.ctrl[motors_servo_id] = np.pi / 2
        data.ctrl[motor_right_thrust_id] = 0.0
        data.ctrl[motor_left_thrust_id] = MAX_THRUST


def keyboard(window, key, scancode, act, mods):
    """
    Updates global state variables based on key presses
    """
    global key_space, key_shift, key_w, key_s, key_a, key_d

    # reset simulation
    if act == glfw.PRESS and key == glfw.KEY_BACKSPACE:
        mj.mj_resetData(model, data)
        mj.mj_forward(model, data)

    # update key states on press and release
    if key == glfw.KEY_SPACE:
        key_space = act != glfw.RELEASE
    elif key == glfw.KEY_LEFT_SHIFT:
        key_shift = act != glfw.RELEASE
    elif key == glfw.KEY_W:
        key_w = act != glfw.RELEASE
    elif key == glfw.KEY_S:
        key_s = act != glfw.RELEASE
    elif key == glfw.KEY_A:
        key_a = act != glfw.RELEASE
    elif key == glfw.KEY_D:
        key_d = act != glfw.RELEASE


def mouse_button(window, button, act, mods):
    # update button state
    global button_left, button_middle, button_right

    button_left = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
    button_middle = (
        glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS
    )
    button_right = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS

    # update mouse position
    glfw.get_cursor_pos(window)


def mouse_move(window, xpos, ypos):
    # compute mouse displacement, save
    global lastx, lasty, button_left, button_middle, button_right

    dx = xpos - lastx
    dy = ypos - lasty
    lastx, lasty = xpos, ypos

    # no buttons down: nothing to do
    if (not button_left) and (not button_middle) and (not button_right):
        return

    # get current window size
    width, height = glfw.get_window_size(window)

    # get shift key state
    PRESS_LEFT_SHIFT = glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
    PRESS_RIGHT_SHIFT = glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
    mod_shift = PRESS_LEFT_SHIFT or PRESS_RIGHT_SHIFT

    # determine action based on mouse button
    if button_right:
        if mod_shift:
            action = mj.mjtMouse.mjMOUSE_MOVE_H
        else:
            action = mj.mjtMouse.mjMOUSE_MOVE_V
    elif button_left:
        if mod_shift:
            action = mj.mjtMouse.mjMOUSE_ROTATE_H
        else:
            action = mj.mjtMouse.mjMOUSE_ROTATE_V
    else:
        action = mj.mjtMouse.mjMOUSE_ZOOM

    mj.mjv_moveCamera(model, action, dx / height, dy / height, scene_main, cam)


def scroll(window, xoffset, yoffset):
    action = mj.mjtMouse.mjMOUSE_ZOOM
    mj.mjv_moveCamera(model, action, 0.0, -0.05 * yoffset, scene_main, cam)


# MuJoCo data structures
model = mj.MjModel.from_xml_path("mochi.xml")  # MuJoCo model
data = mj.MjData(model)  # MuJoCo data
cam = mj.MjvCamera()  # Abstract camera
opt = mj.MjvOption()  # visualization options

# Init GLFW, create window, make OpenGL context current, request v-sync
glfw.init()
window = glfw.create_window(1600, 900, "Mochi Simulation", None, None)
glfw.make_context_current(window)
glfw.swap_interval(1)

# initialize visualization data structures
mj.mjv_defaultCamera(cam)
mj.mjv_defaultOption(opt)
scene_main = mj.MjvScene(model, maxgeom=10000)
scene_pip = mj.MjvScene(model, maxgeom=10000)  # add scene for pip view
context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)

# install GLFW mouse and keyboard callbacks
glfw.set_key_callback(window, keyboard)
glfw.set_cursor_pos_callback(window, mouse_move)
glfw.set_mouse_button_callback(window, mouse_button)
glfw.set_scroll_callback(window, scroll)


# initialize the controller
init_controller(model, data)

# set the controller
mj.set_mjcb_control(controller)

# get the sensor IDs and index ranges
nicla_vision_id = model.camera("nicla_vision").id
ultrasonic_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SENSOR, "ultrasonic")
accelerometer_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SENSOR, "accelerometer")
accelerometer_index_start = model.sensor_adr[accelerometer_id]
accelerometer_index_dim = model.sensor_dim[accelerometer_id]
accelerometer_index_end = accelerometer_index_start + accelerometer_index_dim
gyro_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SENSOR, "gyro")
gyro_index_start = model.sensor_adr[gyro_id]
gyro_index_dim = model.sensor_dim[gyro_id]
gyro_index_end = gyro_index_start + gyro_index_dim
barometer_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SENSOR, "barometer")
barometer_index_start = model.sensor_adr[barometer_id]
barometer_index_dim = model.sensor_dim[barometer_id]
barometer_index_end = barometer_index_start + barometer_index_dim


while not glfw.window_should_close(window):
    time_prev = data.time

    while data.time - time_prev < 1.0 / 60.0:
        mj.mj_step(model, data)

    # update scene and render
    viewport_width, viewport_height = glfw.get_framebuffer_size(window)
    viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

    mj.mjv_updateScene(
        model, data, opt, None, cam, mj.mjtCatBit.mjCAT_ALL.value, scene_main
    )
    mj.mjr_render(viewport, scene_main, context)

    # update scene and render for pip view
    pip_width, pip_height = 320, 240
    main_width, main_height = viewport_width, viewport_height
    pip_x = main_width - pip_width - 20  # 20 px margin from right
    pip_y = 20  # 20 px margin from bottom
    pip_viewport = mj.MjrRect(pip_x, pip_y, pip_width, pip_height)
    pip_cam = mj.MjvCamera()
    pip_cam.type = mj.mjtCamera.mjCAMERA_FIXED
    pip_cam.fixedcamid = nicla_vision_id
    mj.mjv_updateScene(
        model,
        data,
        opt,
        None,
        pip_cam,
        mj.mjtCatBit.mjCAT_ALL.value,
        scene_pip,
    )
    mj.mjr_render(pip_viewport, scene_pip, context)

    # get sensor data
    accelerometer_data = data.sensordata[
        accelerometer_index_start:accelerometer_index_end
    ]
    accelerometer_data[2] -= 9.81  # remove gravity
    gyro_data = data.sensordata[gyro_index_start:gyro_index_end]
    ultrasonic_data = data.sensordata[model.sensor_adr[ultrasonic_id]]
    barometer_data = data.sensordata[barometer_index_end - 1]  # just the z value

    sensor_data_formatted = (
        f"Acel: {accelerometer_data[0]:8.3f}, {accelerometer_data[1]:8.3f}, {accelerometer_data[2]:8.3f} m/s^2\n"
        f"Gyro: {gyro_data[0]:8.3f}, {gyro_data[1]:8.3f}, {gyro_data[2]:8.3f} rad/s\n"
        f"Ultra: {ultrasonic_data:8.3f} m\n"
        f"Baro: {barometer_data:8.3f} m"
    )
    mj.mjr_overlay(
        mj.mjtFont.mjFONT_NORMAL,
        mj.mjtGridPos.mjGRID_TOPLEFT,
        viewport,
        sensor_data_formatted,
        None,
        context,
    )

    # swap OpenGL buffers (blocking call due to v-sync)
    glfw.swap_buffers(window)

    # process pending GUI events, call GLFW callbacks
    glfw.poll_events()

glfw.terminate()

import time
import pygame
import queue as pyqueue
import platform

pygame.init()

def joystick_process_main(output_queue, stop_event):
    # 在子进程中初始化 pygame 并读取手柄状态
    # 允许在后台（无焦点/无窗口）接收手柄事件
    import os
    os.environ["SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS"] = "1"
    # 使用 dummy 显示驱动，避免创建窗口以及依赖 Cocoa 主线程事件循环
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    # 初始化显示模块以启用事件系统（即使用 dummy 驱动）
    try:
        pygame.display.init()
    except Exception:
        pass
    pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        print("未检测到手柄设备")
        return
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    print(f"已连接手柄: {joystick.get_name()}")
    try:
        while not stop_event.is_set():
            # 消费事件队列，确保 SDL 更新摇杆状态
            for _ in pygame.event.get():
                pass
            axes = [joystick.get_axis(i) for i in range(joystick.get_numaxes())]
            buttons = [joystick.get_button(i) for i in range(joystick.get_numbuttons())]
            if platform.system() != "darwin":
                buttons += [0, 0, 0, 0]  # 预留4个位置给方向键
                hat_states = [joystick.get_hat(i) for i in range(joystick.get_numhats())]
                if hat_states[0][1] == 1:
                    buttons[11] = 1
                    buttons[12] = 0
                elif hat_states[0][1] == -1:
                    buttons[11] = 0
                    buttons[12] = 1
                else:
                    buttons[11] = 0
                    buttons[12] = 0
                if hat_states[0][0] == 1:
                    buttons[13] = 0
                    buttons[14] = 1
                elif hat_states[0][0] == -1:
                    buttons[13] = 1
                    buttons[14] = 0
                else:
                    buttons[13] = 0
                    buttons[14] = 0
            
            try:
                if output_queue.full():
                    try:
                        output_queue.get_nowait()
                    except pyqueue.Empty:
                        pass
                output_queue.put_nowait((axes, buttons))
            except pyqueue.Full:
                pass
            time.sleep(1.0 / 120.0)
    finally:
        try:
            pygame.quit()
        except Exception:
            pass


if __name__ == "__main__":
    import numpy as np
    np.set_printoptions(precision=3, suppress=True, linewidth=300)

    # 初始化pygame
    pygame.joystick.init()

    # 检查是否有手柄连接
    if pygame.joystick.get_count() == 0:
        print("未检测到手柄设备")
        exit()

    # 获取第一个手柄
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    print(f"已连接手柄: {joystick.get_name()}")

    try:
        while True:
            # 处理事件队列
            for event in pygame.event.get():
                if event.type == pygame.JOYAXISMOTION:
                    # 获取所有轴的数据
                    axes = [joystick.get_axis(i) for i in range(joystick.get_numaxes())]
                    print("摇杆数据:", np.array(axes))
                elif event.type == pygame.JOYBUTTONDOWN or event.type == pygame.JOYBUTTONUP or event.type == pygame.JOYHATMOTION:
                    # 获取所有按键状态
                    buttons = [joystick.get_button(i) for i in range(joystick.get_numbuttons())]
                    # for i in range(joystick.get_numhats()):
                    #     buttons += list(joystick.get_hat(i))
                    if platform.system() != "darwin":
                        buttons += [0, 0, 0, 0]  # 预留4个位置给方向键
                        hat_states = [joystick.get_hat(i) for i in range(joystick.get_numhats())]
                        if hat_states[0][0] == 1:
                            buttons[11] = 0
                            buttons[12] = 1
                        elif hat_states[0][0] == -1:
                            buttons[11] = 1
                            buttons[12] = 0
                        else:
                            buttons[11] = 0
                            buttons[12] = 0
                        if hat_states[0][1] == 1:
                            buttons[13] = 0
                            buttons[14] = 1
                        elif hat_states[0][1] == -1:
                            buttons[13] = 1
                            buttons[14] = 0
                        else:
                            buttons[13] = 0
                            buttons[14] = 0
                    print("按键状态:", buttons)
    except KeyboardInterrupt:
        print("程序结束")
    finally:
        pygame.quit()
import numpy as np


class KeyboardCommandAdapter:
    def __init__(self):
        self.command = np.zeros(3, dtype=np.float32)

    def update_from_input(self, input) -> None:
        if input.is_key_pressed("up") or input.is_key_pressed("w"):
            self.command[0] = 1.0
        elif input.is_key_pressed("down") or input.is_key_pressed("s"):
            self.command[0] = -1.0
        else:
            self.command[0] = 0.0

        if input.is_key_pressed("left"):
            self.command[1] = 0.5
        elif input.is_key_pressed("right"):
            self.command[1] = -0.5
        else:
            self.command[1] = 0.0

        if input.is_key_pressed("a"):
            self.command[2] = 2.0
        elif input.is_key_pressed("d"):
            self.command[2] = -2.0
        else:
            self.command[2] = 0.0

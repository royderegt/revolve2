from pyrr import Vector3

from .._right_angles import RightAngles
from ..base import Brick


class BrickV2Large(Brick):
    """A brick module for a modular robot."""

    def __init__(self, rotation: float | RightAngles, bone_length: float):
        """
        Initialize this object.

        :param rotation: The modules' rotation.
        """
        super().__init__(
            rotation=rotation,
            bounding_box=Vector3([bone_length, 0.075, 0.075]),
            mass=(42.65 + (bone_length-0.075) * 0.44531428571) / 1000, # Might need to update later
            front_offset= bone_length / 2.0,
            side_offset= 0.075 / 2.0,
            sensors=[],
        )

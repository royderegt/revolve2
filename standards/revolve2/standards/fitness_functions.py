"""Standard fitness functions for modular robots."""

import math

from revolve2.modular_robot_simulation import ModularRobotSimulationState
from revolve2.modular_robot import ModularRobot


from revolve2.modular_robot.body import Module
from revolve2.modular_robot.body.base import ActiveHinge, Core


def xy_displacement(
    begin_state: ModularRobotSimulationState, end_state: ModularRobotSimulationState
) -> float:
    """
    Calculate the distance traveled on the xy-plane by a single modular robot.

    :param begin_state: Begin state of the robot.
    :param end_state: End state of the robot.
    :returns: The calculated fitness.
    """
    begin_position = begin_state.get_pose().position
    end_position = end_state.get_pose().position
    return math.sqrt(
        (begin_position.x - end_position.x) ** 2
        + (begin_position.y - end_position.y) ** 2
    )

def xyz_displacement(
    begin_state: ModularRobotSimulationState, end_state: ModularRobotSimulationState
) -> float:
    """
    Calculate the distance traveled on the xy-plane by a single modular robot.

    :param begin_state: Begin state of the robot.
    :param end_state: End state of the robot.
    :returns: The calculated fitness.
    """
    begin_position = begin_state.get_pose().position
    end_position = end_state.get_pose().position
    return math.sqrt(
        (begin_position.x - end_position.x) ** 2
        + (begin_position.y - end_position.y) ** 2 +
        (begin_position.z - end_position.z) ** 2
    )

def xy_displacement_penalize_z_displacement(
    begin_state: ModularRobotSimulationState, end_state: ModularRobotSimulationState
) -> float:
    """
    Calculate the distance traveled on the xy-plane by a single modular robot.

    :param begin_state: Begin state of the robot.
    :param end_state: End state of the robot.
    :returns: The calculated fitness.
    """
    begin_position = begin_state.get_pose().position
    end_position = end_state.get_pose().position

    if (begin_position.z - end_position.z) > 0:
        penalty = (begin_position.z - end_position.z) * 2
    else:
        penalty = 0
    return math.sqrt(
        (begin_position.x - end_position.x) ** 2
        + (begin_position.y - end_position.y) ** 2
    ) - penalty

def z_value(
    begin_state: ModularRobotSimulationState, end_state: ModularRobotSimulationState
) -> float:
    """
    Calculate the distance traveled on the xy-plane by a single modular robot.

    :param begin_state: Begin state of the robot.
    :param end_state: End state of the robot.
    :returns: The calculated fitness.
    """
    end_position = end_state.get_pose().position

    return end_position.z ** 2

def z_value_xy_displacement(
    begin_state: ModularRobotSimulationState, end_state: ModularRobotSimulationState
) -> float:
    return z_value(begin_state, end_state) + xy_displacement(begin_state, end_state)

def z_value_z_displacement(
    begin_state: ModularRobotSimulationState, end_state: ModularRobotSimulationState
) -> float:
    begin_position = begin_state.get_pose().position
    end_position = end_state.get_pose().position
    delta_z = begin_position.z - end_position.z
    if delta_z < 0:
        penalty = delta_z
    else:
        penalty = 0
    return z_value(begin_state, end_state) - penalty

def z_value_xy_displacement_z_displacement(
    begin_state: ModularRobotSimulationState, end_state: ModularRobotSimulationState
) -> float:
    return z_value_z_displacement(begin_state, end_state) + xy_displacement(begin_state, end_state)

def target_z_value(
    begin_state: ModularRobotSimulationState, end_state: ModularRobotSimulationState, target_z: float
) -> float:
    """
    Calculate the fitness based on how close the final Z position is to the target Z value.

    :param begin_state: Begin state of the robot.
    :param end_state: End state of the robot.
    :param target_z: The target Z value to achieve.
    :returns: The calculated fitness.
    """
    end_position = end_state.get_pose().position
    z_distance = abs(end_position.z - target_z)
    fitness = 1 / (1 + z_distance)
    return fitness

def weighted_target_z_and_xy_displacement(
    begin_state: ModularRobotSimulationState, end_state: ModularRobotSimulationState, target_z: float, z_weight: float, xy_weight: float
) -> float:
    """
    Calculate the fitness based on how close the final Z position is to the target Z value and the XY displacement.

    :param begin_state: Begin state of the robot.
    :param end_state: End state of the robot.
    :param target_z: The target Z value to achieve.
    :param z_weight: The weight for the Z value fitness.
    :param xy_weight: The weight for the XY displacement fitness.
    :returns: The calculated fitness.
    """
    end_position = end_state.get_pose().position
    z_distance = abs(end_position.z - target_z)
    z_fitness = 1 / (1 + z_distance)

    if z_distance > 0.1: z_fitness = 0

    begin_position = begin_state.get_pose().position
    xy_displacement = math.sqrt(
        (begin_position.x - end_position.x) ** 2
        + (begin_position.y - end_position.y) ** 2
    )
    xy_fitness = xy_displacement

    fitness = (z_weight * z_fitness) + (xy_weight * xy_fitness)
    return fitness

def combined_fitness(
    robot: ModularRobot, begin_state: ModularRobotSimulationState, end_state: ModularRobotSimulationState, target_z: float, z_weight: float, xy_weight: float
) -> float:
    """
    Calculate the fitness based on how close the final Z position is to the target Z value and the XY displacement.

    :param begin_state: Begin state of the robot.
    :param end_state: End state of the robot.
    :param target_z: The target Z value to achieve.
    :param z_weight: The weight for the Z value fitness.
    :param xy_weight: The weight for the XY displacement fitness.
    :returns: The calculated fitness.
    """
    active_hinges = len(robot.body.find_modules_of_type(ActiveHinge))
    modules = len(robot.body.find_modules_of_type(Module, exclude=[Core]))

    # Jan 21: added proportional penalty
    if modules <= 4:
        proportion = 0.5
    else:
        proportion = active_hinges / (modules - 4)

    if proportion < 0.25 or proportion > 0.70:
        proportion_penalty = 0.5
    else:
        proportion_penalty = 1

    begin_position = begin_state.get_pose().position
    end_position = end_state.get_pose().position

    if (begin_position.z - end_position.z) > 0:
        falling_penalty = 0.5
    else:
        falling_penalty = 1
    return (proportion_penalty * falling_penalty *
            weighted_target_z_and_xy_displacement(begin_state, end_state, target_z, z_weight, xy_weight))

def split_fitness(
    begin_state: ModularRobotSimulationState, end_state: ModularRobotSimulationState, target_z: float, z_weight: float,
        xy_weight: float, generation_number: int, generation_threshold: int
) -> float:
    end_position = end_state.get_pose().position
    begin_position = begin_state.get_pose().position

    if (begin_position.z - end_position.z) > 0:
        penalty = (begin_position.z - end_position.z) * 2
    else:
        penalty = 0

    if generation_number < generation_threshold:
        z_distance = abs(end_position.z - target_z)
        fitness = 1 / (1 + z_distance)
        return fitness - penalty

    return math.sqrt(
            (begin_position.x - end_position.x) ** 2
            + (begin_position.y - end_position.y) ** 2) - penalty
"""Evaluator class."""
from typing import List, Any

from database_components import Genotype

from revolve2.experimentation.evolution.abstract_elements import Evaluator as Eval
from revolve2.modular_robot_simulation import (
    ModularRobotScene,
    Terrain,
    simulate_scenes,
)
from revolve2.simulators.mujoco_simulator import LocalSimulator
from revolve2.standards import fitness_functions, terrains
from revolve2.standards.simulation_parameters import make_standard_batch_parameters
from revolve2.modular_robot.body.v2 import BodyV2, BrickV2Large


class Evaluator(Eval):
    """Provides evaluation of robots."""

    _simulator: LocalSimulator
    _terrain: Terrain

    def __init__(
        self,
        headless: bool,
        num_simulators: int,
    ) -> None:
        """
        Initialize this object.

        :param headless: `headless` parameter for the physics simulator.
        :param num_simulators: `num_simulators` parameter for the physics simulator.
        """
        self._simulator = LocalSimulator(
            headless=headless, num_simulators=num_simulators
        )
        self._terrain = terrains.flat()

    def evaluate(
        self,
        population: list[Genotype],
        generation_index: int = 0,
    ) -> list[float]:
        """
        Evaluate multiple robots.

        Fitness is the distance traveled on the xy plane.

        :param population: The robots to simulate.
        :param generation_index: The index of the generation.
        :returns: Fitnesses of the robots.
        """
        robots = [genotype.develop() for genotype in population]

        # Create the scenes.
        scenes = []
        for robot in robots:
            scene = ModularRobotScene(terrain=self._terrain)
            scene.add_robot(robot)
            scenes.append(scene)

        # Simulate all scenes.
        scene_states = simulate_scenes(
            simulator=self._simulator,
            batch_parameters=make_standard_batch_parameters(),
            scenes=scenes,
        )

        # Calculate the xy displacements.

        xy_displacements = [
        fitness_functions.combined_fitness(
            robot,
            states[0].get_modular_robot_simulation_state(robot),
            states[-1].get_modular_robot_simulation_state(robot),
            0.5,
            1,
            1
            )
            for robot, states in zip(robots, scene_states)
        ]



        # xy_displacements = [
        # fitness_functions.split_fitness(
        #     states[0].get_modular_robot_simulation_state(robot),
        #     states[-1].get_modular_robot_simulation_state(robot),
        #     0.5,
        #     1,
        #     1,
        #     generation_index,
        #     500
        #     )
        #     for robot, states in zip(robots, scene_states)
        # ]

        return xy_displacements

    def evaluate_positions(
        self,
        population: list[Genotype],
        generation_index: int = 0,
    ) -> list[list[Any]]:
        """
        Evaluate multiple robots.

        Fitness is the distance traveled on the xy plane.

        :param population: The robots to simulate.
        :param generation_index: The index of the generation.
        :returns: Fitnesses of the robots.
        """
        robots = [genotype.develop() for genotype in population]

        # Create the scenes.
        scenes = []
        for robot in robots:
            scene = ModularRobotScene(terrain=self._terrain)
            scene.add_robot(robot)
            scenes.append(scene)

        # Simulate all scenes.
        scene_states = simulate_scenes(
            simulator=self._simulator,
            batch_parameters=make_standard_batch_parameters(),
            scenes=scenes,
        )

        positions = []
        for robot, states in zip(robots, scene_states):
            positions.append([state.get_modular_robot_simulation_state(robot).get_pose().position for state in states])
        # Calculate the xy displacements.

        return positions

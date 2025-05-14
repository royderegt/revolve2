"""Rerun the best robot between all experiments."""

import logging
import statistics

import config
import matplotlib.pyplot as plt
import pandas as pd

from database_components import Genotype, Individual, Generation
from evaluator import Evaluator
from sqlalchemy import select
from sqlalchemy.orm import Session

from revolve2.experimentation.database import OpenMethod, open_database_sqlite
from revolve2.experimentation.experiment_logging import setup_logging

from revolve2.modular_robot.body import Module
from revolve2.modular_robot.body.base import ActiveHinge, Core
from revolve2.modular_robot.body.v2 import BrickV2Large


def main() -> None:
    """Perform the rerun."""
    setup_logging()

    # Load the best individual from the database.
    dbengine = open_database_sqlite(
        config.DATABASE_FILE, open_method=OpenMethod.OPEN_IF_EXISTS
    )

    with Session(dbengine) as ses:
        rows = ses.execute(
            select(Genotype, Individual.fitness)
            .join_from(Genotype, Individual, Genotype.id == Individual.genotype_id)
            .order_by(Individual.fitness.desc())
            .limit(10)

        ).all()
        assert rows is not None

    # Create the evaluator.
    evaluator = Evaluator(headless=False, num_simulators=1)

    # Show the robot.
    for genotype, fitness in rows:
        logging.info(f"Fitness: {fitness}")
        evaluator.evaluate([genotype])



if __name__ == "__main__":
    main()

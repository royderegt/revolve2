"""Rerun the best robot between all experiments."""

import logging

import config
from database_components import Genotype, Individual
from evaluator import Evaluator
from sqlalchemy import select
from sqlalchemy.orm import Session

from revolve2.experimentation.database import OpenMethod, open_database_sqlite
from revolve2.experimentation.experiment_logging import setup_logging


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
            .limit(100)
        ).all()
        assert rows is not None

        genotypes = [row[0] for row in rows]
        #fitnesses = [row[1] for row in rows]

    #logging.info(f"Best fitness: {fitnesses[1]}")

    # Create the evaluator.
    evaluator = Evaluator(headless=False, num_simulators=1)

    # Show the robot.
    evaluator.evaluate(genotypes)


if __name__ == "__main__":
    main()

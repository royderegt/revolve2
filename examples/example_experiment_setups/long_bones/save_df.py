"""Plot fitness over generations for all experiments, averaged."""

import config
import matplotlib.pyplot as plt
import pandas
from database_components import Experiment, Generation, Individual, Population
from sqlalchemy import select

from revolve2.experimentation.database import OpenMethod, open_database_sqlite
from revolve2.experimentation.experiment_logging import setup_logging


def main() -> None:
    """Run the program."""
    setup_logging()

    dbengine = open_database_sqlite(
        "final_runs/5rep_split_1000gen_regular.sqlite", open_method=OpenMethod.OPEN_IF_EXISTS
    )

    df = pandas.read_sql(
        select(
            Experiment.id.label("experiment_id"),
            Generation.generation_index,
            Individual.fitness,
        )
        .join_from(Experiment, Generation, Experiment.id == Generation.experiment_id)
        .join_from(Generation, Population, Generation.population_id == Population.id)
        .join_from(Population, Individual, Population.id == Individual.population_id),
        dbengine,
    )

    agg_per_experiment_per_generation = (
        df.groupby(["experiment_id", "generation_index"])
        .agg({"fitness": ["max", "mean"]})
        .reset_index()
    )
    agg_per_experiment_per_generation.columns = [
        "experiment_id",
        "generation_index",
        "max_fitness",
        "mean_fitness",
    ]

    agg_per_generation = (
        agg_per_experiment_per_generation.groupby("generation_index")
        .agg({"max_fitness": ["mean", "std"], "mean_fitness": ["mean", "std"]})
        .reset_index()
    )
    agg_per_generation.columns = [
        "generation_index",
        "max_fitness_mean",
        "max_fitness_std",
        "mean_fitness_mean",
        "mean_fitness_std",
    ]

    df.to_csv("./data_maarten/split_regular.csv")

if __name__ == "__main__":
    main()

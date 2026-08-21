from __future__ import annotations

import pandas as pd
import pytest

from animal_intervention.experiments.preparedness_value_decomposition import decompose_path


def test_path_components_telescope_to_total_opportunity() -> None:
    row = {
        "dataset_id": "d",
        "network_id": "n",
        "anchor_id": "a",
        "parameter_id": "p",
        "epidemic_model": "temporal_sir",
        "random_block": 0,
        "initial_infected": "x",
        "world_seed": 1,
        "system_family": "f",
        "analysis_cluster_id": "c",
        "population_size": 10,
        "random__case_only": 9,
        "history_weight__case_only": 8,
        "history_coverage__case_only": 8,
        "history_coverage__random": 6,
        "history_coverage__history_weight": 5,
        "full_surveillance__history_weight": 3,
    }
    components = decompose_path(pd.DataFrame([row]))
    assert components["value"].sum() == pytest.approx(0.6)
    assert components.loc[components["component"].eq("response_targeting"), "value"].iloc[0] == pytest.approx(0.1)

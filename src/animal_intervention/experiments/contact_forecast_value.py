from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import platform
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import yaml

from animal_intervention.evaluation import stable_hash_order

from .historical_set_planning import (
    _all_subsets,
    _candidate_pool,
    _checkpoint_key,
    _choose_sets,
    _even_windows,
    _history_only_policy_scores,
    _input_hashes,
    _score_history_sets,
)
from .history_baseline_substitution import _markdown_table
from .immediate_case_targeting import _case_conditioned_history_sets
from .intervention_delivery_sensitivity import (
    SYSTEM_FAMILY_LABELS,
    _hierarchical_summary,
    _parameter_pool,
    _select_parameter_regimes,
)
from .outbreak_response_pilot import (
    _git_value,
    _keyed_seed,
    _load_source_config,
    _load_windows,
    _matching_stable_scores,
    _sha256,
)
from .role_aware_sentinel_response import _replay_response, _top_history
from .sequential_preparedness_update import _budget, _parameters


KEYS = [
    "dataset_id", "network_id", "system_family", "analysis_cluster_id",
    "anchor_id", "anchor_time", "epidemic_model", "initial_infected",
]


def _planning_budget(population_size: int, decision: dict[str, Any]) -> int:
    """Return the pre-specified intervention-set size for one population."""
    budget = _budget(
        population_size,
        int(decision["minimum_budget"]),
        float(decision["response_budget_fraction"]),
    )
    if population_size >= int(decision["minimum_population_for_pair"]):
        budget = max(budget, 2)
    return min(int(decision["maximum_planning_budget"]), budget)


def _expanded_pool(
    task: dict[str, Any], config: dict[str, Any], initial: str, budget: int
) -> tuple[str, ...]:
    anchor = task["window"]["anchor"]
    eligible = set(map(str, task["window"]["eligible"]))
    mean_period = pd.Timedelta(days=float(task["parameter"].mean_infectious_period_days))
    base, _, _ = _candidate_pool(
        window=task["window"], stable_scores=task["stable_scores"], eligible=eligible,
        initial=initial, pool_per_signal=budget, mean_period=mean_period,
        seed=_keyed_seed(int(config["evaluation"]["seed"]), task["dataset_id"], anchor.anchor_id, initial, "forecast_pool"),
    )
    order = stable_hash_order(
        sorted(eligible - {initial} - set(base)), int(config["evaluation"]["seed"]),
        task["dataset_id"], task["network_id"], anchor.anchor_id, initial, "exploration_candidates",
    )
    expanded = [*base, *order[: int(config["decision"]["exploration_candidates"])]]
    return tuple(expanded[: int(config["decision"]["maximum_candidate_pool"])])


def _future_set_values(
    task: dict[str, Any], config: dict[str, Any], initial: str,
    pool: tuple[str, ...], budget: int,
) -> pd.DataFrame:
    anchor = task["window"]["anchor"]; start = pd.Timestamp(anchor.anchor_time); end = pd.Timestamp(anchor.horizon_end)
    mean_period = pd.Timedelta(days=float(task["parameter"].mean_infectious_period_days))
    engine, parameters = _parameters(task["parameter"], task["model"], mean_period)
    population = len(task["window"]["future"].nodes()); rows = []
    subsets = [item for item in _all_subsets(pool, budget) if len(item) == budget]
    total_blocks = int(task["selection_blocks"]) + int(task["evaluation_blocks"])
    for block in range(total_blocks):
        world_seed = _keyed_seed(int(config["evaluation"]["seed"]), task["dataset_id"], task["network_id"], anchor.anchor_id, task["model"]["name"], initial, "forecast_future", block)
        natural = engine.simulate(task["window"]["future"], parameters, initial_infected=(initial,), start_time=start, end_time=end, world_seed=world_seed)
        case, _ = _replay_response(engine=engine, parameters=parameters, future=task["window"]["future"], natural=natural, initial=initial, world_seed=world_seed, start_time=start, end_time=end, detection_time=start, action_delay=pd.Timedelta(0), targets={initial}, residual=float(config["decision"]["residual_contact_multiplier"]))
        for subset in subsets:
            result, _ = _replay_response(engine=engine, parameters=parameters, future=task["window"]["future"], natural=natural, initial=initial, world_seed=world_seed, start_time=start, end_time=end, detection_time=start, action_delay=pd.Timedelta(0), targets={initial, *subset}, residual=float(config["decision"]["residual_contact_multiplier"]))
            rows.append({
                "future_block": block, "set_signature": "|".join(subset),
                "value": (case.final_size-result.final_size)/population,
                "case_only_final_size": case.final_size, "population_size": population,
            })
    return pd.DataFrame(rows)


def _select_signature(table: pd.DataFrame, blocks: list[int]) -> str:
    means = table.loc[table.future_block.isin(blocks)].groupby("set_signature", observed=True).value.mean().reset_index()
    return str(means.sort_values(["value", "set_signature"], ascending=[False, True]).iloc[0].set_signature)


def _run_task(task: dict[str, Any], config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    anchor = task["window"]["anchor"]; eligible = set(map(str, task["window"]["eligible"])); rows=[]; diagnostics=[]
    for initial in task["seeds"]:
        initial = str(initial)
        budget = min(
            _planning_budget(len(eligible), config["decision"]),
            len(eligible - {initial}),
        )
        pool = _expanded_pool(task, config, initial, budget); budget=min(budget,len(pool))
        if budget == 0: continue
        history = _score_history_sets(task=task, config=config, initial=initial, pool=pool, budget=budget)
        history_exact, history_singleton, history_means = _choose_sets(history, pool, budget, task["stable_scores"])
        stable = _top_history(task["stable_scores"], eligible, budget, _keyed_seed(int(config["evaluation"]["seed"]), "forecast_stable", initial), excluded={initial})
        mean_period = pd.Timedelta(days=float(task["parameter"].mean_infectious_period_days))
        ring = dict(_case_conditioned_history_sets(history_stream=task["window"]["history"], stable_scores=task["stable_scores"], eligible=eligible, initial=initial, budget=budget, history_start=pd.Timestamp(anchor.history_start), anchor_time=pd.Timestamp(anchor.anchor_time), recency_half_life=mean_period, seed=_keyed_seed(int(config["evaluation"]["seed"]), "forecast_ring", initial)))["past_weight_ring"]
        future = _future_set_values(task, config, initial, pool, budget)
        blocks = sorted(future.future_block.unique())
        split = int(task["selection_blocks"])
        selection_blocks = blocks[:split]
        evaluation_blocks = blocks[split:]
        forecast_signature=_select_signature(future,selection_blocks); forecast=set(forecast_signature.split("|"))
        methods={"future_contact_oracle":forecast,"history_exact":history_exact,"history_singleton":history_singleton,"stable":stable,"static_ring":ring}
        common={"dataset_id":task["dataset_id"],"network_id":task["network_id"],"system_family":task["system_family"],"analysis_cluster_id":task["analysis_cluster_id"],"anchor_id":anchor.anchor_id,"anchor_time":pd.Timestamp(anchor.anchor_time),"epidemic_model":task["model"]["name"],"initial_infected":initial,"budget":budget}
        eval_future=future.loc[future.future_block.isin(evaluation_blocks)]
        for method,nodes in methods.items():
            signature="|".join(sorted(nodes)); selected=eval_future.loc[eval_future.set_signature.eq(signature)]
            if selected.empty:
                raise AssertionError(f"method set absent from candidate pool: {method} {signature} {pool}")
            for row in selected.itertuples(index=False): rows.append({**common,"future_block":int(row.future_block),"method":method,"selected_nodes":signature,"value":float(row.value)})
        history_target=history_means.loc[history_means.set_size.eq(budget),["set_signature","value"]].rename(columns={"value":"history_value"})
        forecast_target=future.loc[future.future_block.isin(selection_blocks)].groupby("set_signature",observed=True).value.mean().reset_index(name="forecast_value")
        aligned=history_target.merge(forecast_target,on="set_signature",how="inner"); rho=aligned.history_value.rank().corr(aligned.forecast_value.rank()) if aligned.history_value.nunique()>1 and aligned.forecast_value.nunique()>1 else np.nan
        diagnostics.append({**common,"candidate_pool":"|".join(pool),"history_choice":"|".join(sorted(history_exact)),"forecast_choice":forecast_signature,"history_forecast_rank_correlation":rho,"forecast_matches_history":forecast==history_exact,"candidate_sets":len(aligned)})
    return pd.DataFrame(rows),pd.DataFrame(diagnostics)


def _contrasts(worlds: pd.DataFrame) -> pd.DataFrame:
    keys=[*KEYS,"budget","future_block"]; wide=worlds.pivot(index=keys,columns="method",values="value").reset_index(); rows=[]
    for name,left,right in [
        ("forecast_oracle_vs_history","future_contact_oracle","history_exact"),
        ("forecast_oracle_vs_stable","future_contact_oracle","stable"),
        ("forecast_oracle_vs_ring","future_contact_oracle","static_ring"),
        ("history_vs_stable","history_exact","stable"),
    ]:
        frame=wide[keys].copy();frame["contrast"]=name;frame["value"]=wide[left]-wide[right];rows.append(frame)
    return pd.concat(rows,ignore_index=True)


def _model_interaction(
    family: pd.DataFrame, bootstrap_replicates: int, seed: int
) -> pd.DataFrame:
    """Summarize the family-level SEIR-minus-SIR contrast without pooling families."""
    wide = family.pivot(
        index=["contrast", "system_family"],
        columns="epidemic_model",
        values="mean_value",
    ).reset_index()
    wide["seir_minus_sir"] = (
        wide["temporal_seir_erlang"] - wide["temporal_sir"]
    )
    rows = []
    for index, (contrast, group) in enumerate(wide.groupby("contrast", sort=True)):
        values = group["seir_minus_sir"].to_numpy(float)
        rng = np.random.default_rng(seed + index)
        samples = np.array(
            [rng.choice(values, len(values), replace=True).mean()
             for _ in range(bootstrap_replicates)]
        )
        rows.append({
            "contrast": contrast,
            "families": len(values),
            "family_equal_mean": float(values.mean()),
            "ci_low": float(np.quantile(samples, 0.025)),
            "ci_high": float(np.quantile(samples, 0.975)),
            "bootstrap_probability_positive": float((samples > 0).mean()),
            "positive_families": int((values > 0).sum()),
        })
    return pd.DataFrame(rows)


def _leave_one_family_out(family: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Expose sensitivity to each independent animal-system family."""
    detail = []
    for (model, contrast), group in family.groupby(
        ["epidemic_model", "contrast"], observed=True, sort=True
    ):
        families = sorted(group["system_family"].unique())
        for omitted in families:
            retained = group.loc[group["system_family"].ne(omitted), "mean_value"]
            detail.append({
                "epidemic_model": model,
                "contrast": contrast,
                "omitted_family": omitted,
                "retained_families": len(retained),
                "family_equal_mean": float(retained.mean()),
            })
    detail_frame = pd.DataFrame(detail)
    summary = (
        detail_frame.groupby(["epidemic_model", "contrast"], observed=True)
        .family_equal_mean.agg(["min", "max"])
        .reset_index()
        .rename(columns={"min": "minimum_leave_one_out_mean", "max": "maximum_leave_one_out_mean"})
    )
    summary["all_leave_one_out_positive"] = summary.minimum_leave_one_out_mean > 0
    return detail_frame, summary


def _plot(summary: pd.DataFrame, diagnostics: pd.DataFrame, path: Path, dpi: int) -> None:
    contrasts=["forecast_oracle_vs_history","forecast_oracle_vs_stable","forecast_oracle_vs_ring"];labels=["Contact-informed vs history planner","Contact-informed vs stable","Contact-informed vs case ring"];models=["temporal_sir","temporal_seir_erlang"]
    fig,axes=plt.subplots(1,2,figsize=(15.5,6.6))
    for y,(contrast,label) in enumerate(zip(contrasts,labels)):
        for offset,model,color,model_label in [(-.12,models[0],"#4C78A8","SIR"),(.12,models[1],"#F58518","SEIR/Erlang")]:
            row=summary.loc[summary.contrast.eq(contrast)&summary.epidemic_model.eq(model)].iloc[0];mean,low,high=100*row[["family_equal_mean","ci_low","ci_high"]].to_numpy(float);axes[0].errorbar(mean,y+offset,xerr=[[mean-low],[high-mean]],fmt="o",color=color,capsize=4,label=model_label if y==0 else None)
    axes[0].axvline(0,color="#555",linestyle="--");axes[0].set_yticks(range(3),labels);axes[0].invert_yaxis();axes[0].set_xlabel("Held-out avoided attack-rate gain (percentage points)");axes[0].set_title("Value of knowing the future contact sequence",weight="bold");axes[0].legend(frameon=False)
    eligible=diagnostics.loc[diagnostics.budget.gt(1)]; grouped=eligible.groupby(["system_family","epidemic_model"],observed=True).history_forecast_rank_correlation.mean().reset_index();families=sorted(grouped.system_family.unique());y=np.arange(len(families));width=.34
    for offset,model,color,label in [(-width/2,models[0],"#4C78A8","SIR"),(width/2,models[1],"#F58518","SEIR/Erlang")]:
        values=grouped.loc[grouped.epidemic_model.eq(model)].set_index("system_family").reindex(families).history_forecast_rank_correlation;axes[1].barh(y+offset,values,height=width,color=color,label=label)
    for index,family in enumerate(families):
        family_values=grouped.loc[grouped.system_family.eq(family),"history_forecast_rank_correlation"]
        if family_values.isna().all():
            axes[1].text(.02,index,"not estimable (tied set values)",va="center",ha="left",fontsize=9,color="#555")
    axes[1].axvline(0,color="#555",linestyle="--");axes[1].set_yticks(y,[SYSTEM_FAMILY_LABELS.get(f,f) for f in families]);axes[1].set_ylim(len(families)-.5,-.5);axes[1].set_xlim(-1,1);axes[1].set_xlabel("History-to-future set-value rank correlation");axes[1].set_title("Can history identify the future-best set?",weight="bold");axes[1].legend(frameon=False)
    for axis in axes:axis.grid(alpha=.2)
    fig.suptitle("The value of contact forecasting for animal outbreak response",fontsize=19,weight="bold");fig.subplots_adjust(left=.20,right=.98,top=.84,bottom=.14,wspace=.42);fig.savefig(path,dpi=dpi);plt.close(fig)


def run(config_path: Path, profile_name: str) -> dict[str,Any]:
    started=time.perf_counter();config=yaml.safe_load(config_path.read_text(encoding="utf-8"));profile=config["profiles"][profile_name];evaluation=config["evaluation"];stable=pd.read_csv(config["data"]["stable_prediction_path"],dtype={"candidate_id":str,"network_id":str});stable["anchor_time"]=pd.to_datetime(stable.anchor_time,format="mixed")
    results_dir=Path(config["outputs"]["results_root"])/config["experiment"]["id"]/profile_name;report_dir=Path(config["outputs"]["report_root"])/config["experiment"]["id"]/profile_name;checkpoint_dir=results_dir/"checkpoints"
    for directory in [results_dir,report_dir,checkpoint_dir]:directory.mkdir(parents=True,exist_ok=True)
    tasks=[]
    for dataset_id in profile["datasets"]:
        spec=config["data"]["datasets"][dataset_id];source_config=_load_source_config(Path(spec["source_config"]));windows=_load_windows(dataset_id,source_config);default=str(spec.get("network_id","all"));[w.setdefault("network_id",default) for w in windows];available=set(stable.loc[stable.dataset_id.eq(dataset_id),["network_id","anchor_time"]].itertuples(index=False,name=None));fallback_datasets=set(config["data"].get("history_score_fallback_datasets",[]));windows=[w for w in windows if dataset_id in fallback_datasets or (str(w["network_id"]),pd.Timestamp(w["anchor"].anchor_time)) in available];windows=_even_windows(windows,profile.get("max_anchors_per_dataset"));parameters=_parameter_pool(Path(spec["source_results"])/"parameter_selection.csv",str(evaluation["parameter_pool"]))
        for window in windows:
            selected=_select_parameter_regimes(list(parameters.itertuples(index=False)),str(evaluation["parameter_selection_mode"]));
            if len(selected)!=1:continue
            parameter=selected[0][1];network_id=str(window["network_id"]);scores=_matching_stable_scores(stable,dataset_id,network_id,window["anchor"].anchor_time,window["eligible"]) if (network_id,pd.Timestamp(window["anchor"].anchor_time)) in available else _history_only_policy_scores(window["history"],set(map(str,window["eligible"])));seeds=stable_hash_order(list(map(str,window["eligible"])),int(evaluation["seed"]),dataset_id,window["anchor"].anchor_id,"contact_forecast_value")[:int(profile["seeds_per_anchor"])];cluster=f"{dataset_id}::{network_id}" if spec.get("analysis_cluster")=="network" else f"{dataset_id}::{network_id}::{window['anchor'].anchor_id}"
            for model in config["decision"]["epidemic_models"]:tasks.append({"dataset_id":dataset_id,"network_id":network_id,"system_family":spec["system_family"],"analysis_cluster_id":cluster,"window":window,"parameter":parameter,"model":model,"stable_scores":scores,"seeds":seeds,"history_blocks":profile["history_blocks"],"selection_blocks":profile["selection_blocks"],"evaluation_blocks":profile["evaluation_blocks"]})
    worlds=[];diagnostics=[]
    for task in tqdm(tasks,desc="Contact-forecast value",unit="task"):
        checkpoint_key=_checkpoint_key([task["dataset_id"],task["network_id"],task["window"]["anchor"].anchor_id,task["model"]["name"],profile_name],config_path,Path(__file__));checkpoint=checkpoint_dir/f"task_{checkpoint_key}.pkl"
        if checkpoint.exists() and config["execution"].get("resume",True):payload=pd.read_pickle(checkpoint)
        else:payload=_run_task(task,config);pd.to_pickle(payload,checkpoint)
        worlds.append(payload[0]);diagnostics.append(payload[1])
    worlds=pd.concat(worlds,ignore_index=True);diagnostics=pd.concat(diagnostics,ignore_index=True);contrasts=_contrasts(worlds);primary=contrasts.loc[contrasts.budget.gt(1)];summary,family=_hierarchical_summary(primary,value_column="value",group_columns=["epidemic_model","contrast"],bootstrap_replicates=int(profile.get("bootstrap_replicates",evaluation["bootstrap_replicates"])),seed=int(evaluation["seed"]));interaction=_model_interaction(family,int(profile.get("bootstrap_replicates",evaluation["bootstrap_replicates"])),int(evaluation["seed"]));lofo_detail,lofo_summary=_leave_one_family_out(family)
    checks={"all_datasets":set(worlds.dataset_id)==set(profile["datasets"]),"expected_families_full":profile_name!="full" or worlds.system_family.nunique()==int(profile.get("expected_system_families",5)),"five_policy_arms":worlds.groupby([*KEYS,"future_block"],observed=True).method.nunique().eq(5).all(),"selection_evaluation_split":int(profile["selection_blocks"])>=4 and worlds.future_block.nunique()==int(profile["evaluation_blocks"]),"finite_values":np.isfinite(worlds.value).all(),"bounded_values":worlds.value.between(-1,1).all(),"candidate_pool_bounded":diagnostics.candidate_pool.str.split("|").str.len().le(int(config["decision"]["maximum_candidate_pool"])).all(),"complete_leave_one_family_out":len(lofo_detail)==len(family.groupby(["epidemic_model","contrast"],observed=True))*family.system_family.nunique(),"interaction_family_support":profile_name!="full" or diagnostics.loc[(diagnostics.budget.astype(int)>1)&(diagnostics.candidate_sets.astype(int)>1),"system_family"].nunique()>=int(profile.get("minimum_interaction_families",4))}
    audit={"status":"pass" if all(checks.values()) else "fail","checks":{k:bool(v) for k,v in checks.items()},"datasets":worlds.dataset_id.nunique(),"families":worlds.system_family.nunique(),"anchors":worlds[["dataset_id","network_id","anchor_id"]].drop_duplicates().shape[0],"planning_contexts":len(diagnostics),"heldout_policy_evaluations":len(worlds)}
    if audit["status"]!="pass":raise ValueError(audit)
    worlds.to_csv(results_dir/"heldout_policy_worlds.csv.gz",index=False,compression="gzip");diagnostics.to_csv(results_dir/"contact_forecast_diagnostics.csv.gz",index=False,compression="gzip");contrasts.to_csv(results_dir/"heldout_contrasts.csv.gz",index=False,compression="gzip");summary.to_csv(results_dir/"contrast_summary.csv",index=False);family.to_csv(results_dir/"family_contrasts.csv",index=False);interaction.to_csv(results_dir/"model_interaction_summary.csv",index=False);lofo_detail.to_csv(results_dir/"leave_one_family_out.csv",index=False);lofo_summary.to_csv(results_dir/"leave_one_family_out_summary.csv",index=False);(results_dir/"audit.json").write_text(json.dumps(audit,indent=2),encoding="utf-8");(results_dir/"resolved_config.yaml").write_text(yaml.safe_dump({**config,"runtime":{"profile":profile_name,"timestamp_utc":datetime.now(UTC).isoformat()}},sort_keys=False),encoding="utf-8");(results_dir/"run_manifest.json").write_text(json.dumps({"experiment_id":config["experiment"]["id"],"profile":profile_name,"created_at_utc":datetime.now(UTC).isoformat(),"elapsed_seconds":round(time.perf_counter()-started,3),"python":platform.python_version(),"platform":platform.platform(),"git_commit":_git_value(["rev-parse","HEAD"]),"git_worktree_dirty":bool(_git_value(["status","--porcelain"])),"config_path":str(config_path),"config_sha256":_sha256(config_path),"source_sha256":_sha256(Path(__file__)),"input_sha256":_input_hashes(config)},indent=2),encoding="utf-8")
    _plot(summary,diagnostics,report_dir/"contact_forecast_value.png",int(profile["render_dpi"]));display=summary.copy();
    for column in ["family_equal_mean","ci_low","ci_high"]:display[column]*=100
    lofo_display=lofo_summary.copy();lofo_display[["minimum_leave_one_out_mean","maximum_leave_one_out_mean"]]*=100
    (report_dir/"STAGE_REPORT.md").write_text("# Contact-forecast value\n\nA candidate-constrained future-contact oracle selects a fixed-budget set from at most six history-derived or deterministic exploration candidates. It uses 32 paired epidemic random blocks on the known future contact sequence and is evaluated on 8 disjoint random blocks. This is a non-deployable, finite-Monte-Carlo value-of-information benchmark within the restricted action space, not a whole-population ceiling, prospective policy, or field-effect estimate.\n\n"+_markdown_table(display)+"\n\n## Leave-one-family-out sensitivity\n\n"+_markdown_table(lofo_display),encoding="utf-8");print(json.dumps(audit,indent=2));return audit


def main()->None:
    parser=argparse.ArgumentParser(description="Quantify the value of future contact information.");parser.add_argument("--config",type=Path,required=True);parser.add_argument("--profile",choices=["smoke","full"],default="smoke");args=parser.parse_args();run(args.config,args.profile)


if __name__=="__main__":main()

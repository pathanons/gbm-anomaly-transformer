from __future__ import annotations

from pathlib import Path

import main


def test_load_flat_yaml_keeps_none_literal_when_quoted(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "pipeline: log_return",
                'association_mode: "none"',
                "tickers: AAPL,MSFT",
                "window_size: 50,100",
            ]
        ),
        encoding="utf-8",
    )

    config = main.load_flat_yaml(config_path, list_int_keys={"window_size"})

    assert config["association_mode"] == "none"
    assert config["tickers"] == ["AAPL", "MSFT"]
    assert config["window_size"] == [50, 100]


def test_namespace_resolves_log_return_defaults() -> None:
    args = main.namespace_for_gbm(
        {
            "pipeline": "log_return",
            "data_path": "datasets/SP500_event_taxonomy_w{window_size}",
            "tickers": "all",
        },
        window_size=100,
        exp_name="unit_exp",
    )

    assert args.exp_name == "unit_exp"
    assert args.data_path == "datasets/SP500_event_taxonomy_w100"
    assert args.association_mode == "gaussian_log_return"
    assert args.tickers is None


def test_experiment_suite_dry_run_expands_grid_and_thresholds(tmp_path: Path, capsys) -> None:
    config_path = tmp_path / "suite.yaml"
    config_path.write_text(
        "\n".join(
            [
                "pipeline: experiment_suite",
                "exp_name: unit_suite",
                "exp_name_template: unit_h{n_heads}_l{e_layers}_lr{lr_tag}_ep{epochs}_s{seed}",
                "data_path: datasets/SP500_event_taxonomy_w{window_size}",
                "window_size: 100",
                "n_heads_values: 2",
                "e_layers_values: 2",
                "lr_values: 0.00005,0.0001",
                "epoch_values: 1",
                "seed_values: 42",
                "mad_k_values: 9,10",
            ]
        ),
        encoding="utf-8",
    )

    main.run_config(config_path, dry_run=True)
    output = capsys.readouterr().out

    assert "experiment suite: trials=2" in output
    assert "unit_h2_l2_lr0p00005_ep1_s42" in output
    assert "unit_h2_l2_lr0p0001_ep1_s42" in output
    assert "suite threshold k=9" in output
    assert "suite threshold k=10" in output


def test_all_yaml_configs_have_valid_pipeline() -> None:
    for config_path in Path("configs").rglob("*.yaml"):
        config = main.load_flat_yaml(config_path, list_int_keys={"window_size", "score_mav_windows"})
        assert main.pipeline_name(config) in main.PIPELINES

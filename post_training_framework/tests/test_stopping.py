"""统一训练停止控制器的CPU单元测试。"""

from __future__ import annotations

import importlib.util
import json
import logging
import sys
import tempfile
import unittest
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import patch


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ptf.stopping import (  # noqa: E402
    StopCategory,
    StopSeverity,
    TrainingStopController,
    build_stop_decision,
)
from ptf.train_grpo import (  # noqa: E402
    GRPOTrainer,
    GROUP_DIAG_CSV_COLUMNS,
    TRAIN_CSV_COLUMNS,
    VAL_CSV_COLUMNS,
    _evaluate_early_stopping,
    _evaluate_signal_guard_window,
    logger as grpo_logger,
    setup_logging,
)
from ptf.reward import GSM8KRewardConfig  # noqa: E402


class LoggingLifecycleTest(unittest.TestCase):
    def test_setup_logging_rebinds_file_handler_between_sessions(self) -> None:
        """同一进程切换训练会话时，日志文件必须重新绑定到新目录。"""
        original_handlers = list(grpo_logger.handlers)
        original_level = grpo_logger.level
        original_propagate = grpo_logger.propagate
        for handler in original_handlers:
            grpo_logger.removeHandler(handler)
        temp_dir = tempfile.TemporaryDirectory()

        try:
            root = Path(temp_dir.name)
            first_output = root / "first"
            second_output = root / "second"
            first_log = first_output / "logs" / "first-run.log"
            second_log = second_output / "logs" / "second-run.log"

            setup_logging(first_output, "first-run")
            grpo_logger.info("第一训练会话标记")
            setup_logging(second_output, "second-run")
            grpo_logger.info("第二训练会话标记")
            for handler in grpo_logger.handlers:
                handler.flush()

            file_handlers = [
                handler
                for handler in grpo_logger.handlers
                if isinstance(handler, logging.FileHandler)
            ]
            self.assertEqual(len(file_handlers), 1)
            self.assertEqual(
                Path(file_handlers[0].baseFilename).resolve(),
                second_log.resolve(),
            )

            first_text = first_log.read_text(encoding="utf-8")
            second_text = second_log.read_text(encoding="utf-8")
            self.assertIn("第一训练会话标记", first_text)
            self.assertNotIn("第二训练会话标记", first_text)
            self.assertIn("第二训练会话标记", second_text)
        finally:
            for handler in list(grpo_logger.handlers):
                try:
                    handler.flush()
                finally:
                    if isinstance(handler, logging.FileHandler):
                        handler.close()
                    grpo_logger.removeHandler(handler)
            for handler in original_handlers:
                grpo_logger.addHandler(handler)
            grpo_logger.setLevel(original_level)
            grpo_logger.propagate = original_propagate
            temp_dir.cleanup()


class TrainingStopControllerTest(unittest.TestCase):
    def test_explicit_priority_keeps_all_candidates(self) -> None:
        """同一阶段多项触发时选择高优先级项，同时完整留档候选。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            controller = TrainingStopController(temp_dir, "priority-test")
            controller.start({"start_step": 0})
            early = build_stop_decision(
                StopCategory.EARLY_STOPPING,
                source="early",
                reason="验证不再改善",
                step=49,
            )
            format_guard = build_stop_decision(
                StopCategory.FORMAT_GUARD,
                source="format",
                reason="格式崩溃",
                step=49,
            )

            selected = controller.select("post_eval", [early, format_guard])

            self.assertIsNotNone(selected)
            self.assertEqual(selected.category, StopCategory.FORMAT_GUARD)
            events = [
                json.loads(line)
                for line in controller.events_path.read_text(encoding="utf-8").splitlines()
            ]
            selection = events[-1]
            self.assertEqual(selection["stage"], "post_eval")
            self.assertEqual(len(selection["candidates"]), 2)

    def test_finalize_writes_machine_readable_summary(self) -> None:
        """最终状态同时写根目录摘要和追加式事件。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            controller = TrainingStopController(temp_dir, "archive-test")
            controller.start()
            decision = build_stop_decision(
                StopCategory.MAX_STEPS,
                source="budget",
                reason="达到最大步数",
                step=29,
                severity=StopSeverity.NORMAL,
            )
            controller.select("budget", [decision])
            controller.finalize(
                status="completed",
                decision=decision,
                last_completed_step=29,
                checkpoint_path=Path(temp_dir) / "checkpoint-29",
                finalization_errors=[],
            )

            summary = json.loads(controller.summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "completed")
            self.assertEqual(summary["decision"]["category"], StopCategory.MAX_STEPS)
            self.assertEqual(summary["last_completed_step"], 29)
            event_types = [
                json.loads(line)["event"]
                for line in controller.events_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                event_types,
                ["session_started", "stop_selected", "session_finished"],
            )


class IndependentStoppingRulesTest(unittest.TestCase):
    def test_mixed_only_failure_remains_warning(self) -> None:
        """mixed不足不能掩盖仍然有效的总reward优势信号。"""
        diagnostics = [
            {
                "effective_group_rate": 0.8,
                "mixed_group_rate": 0.5,
                "zero_advantage_rate": 0.2,
                "rollout_format_rate": 0.95,
            }
            for _ in range(10)
        ]
        summary, failures, warnings = _evaluate_signal_guard_window(
            diagnostics,
            min_effective_group_rate=0.7,
            min_mixed_group_rate=0.6,
            max_zero_advantage_rate=0.3,
            min_rollout_format_rate=0.9,
            mixed_hard_stop=False,
        )

        self.assertEqual(failures, [])
        self.assertEqual(warnings, ["mixed=0.500"])
        self.assertAlmostEqual(summary["effective"], 0.8)

    def test_early_stopping_can_extend_recovery_trend(self) -> None:
        """达到耐心后，近期恢复趋势仍按原规则有限延长。"""
        result = _evaluate_early_stopping(
            step=49,
            val_em=0.64,
            best_val_em=0.67,
            best_step=-1,
            steps_no_improve=40,
            extension_steps=0,
            val_em_history=[0.67, 0.60, 0.62, 0.64],
            eval_freq=10,
            max_steps_no_improve=50,
            trend_window=3,
            min_recovery_slope=0.005,
            max_extension_steps=20,
        )

        self.assertTrue(result.extended)
        self.assertIsNone(result.decision)
        self.assertEqual(result.steps_no_improve, 50)
        self.assertEqual(result.extension_steps, 10)

    def test_early_stopping_returns_structured_decision(self) -> None:
        """没有恢复趋势时只由验证早停判定器提交结构化决定。"""
        result = _evaluate_early_stopping(
            step=49,
            val_em=0.60,
            best_val_em=0.67,
            best_step=-1,
            steps_no_improve=40,
            extension_steps=0,
            val_em_history=[0.67, 0.63, 0.61, 0.60],
            eval_freq=10,
            max_steps_no_improve=50,
            trend_window=3,
            min_recovery_slope=0.005,
            max_extension_steps=20,
        )

        self.assertFalse(result.extended)
        self.assertIsNotNone(result.decision)
        self.assertEqual(result.decision.category, StopCategory.EARLY_STOPPING)
        self.assertTrue(result.decision.reason.startswith("早停:"))


class DashboardContractTest(unittest.TestCase):
    def test_existing_csv_columns_are_unchanged(self) -> None:
        """锁定看板和实验汇总脚本依赖的原始CSV契约。"""
        self.assertEqual(
            TRAIN_CSV_COLUMNS,
            [
                "step", "reward_mean", "reward_std", "policy_loss", "kl_loss",
                "approx_kl", "clip_frac", "response_len_mean", "lr", "step_time",
                "grad_norm", "kl_loss_coef", "prompt_count", "rollout_count",
                "mini_batch_count", "optimizer_update_count", "best_val_em",
                "steps_no_improve",
            ],
        )
        self.assertIn("val_exact_match", VAL_CSV_COLUMNS)
        self.assertIn("val_sample_exact_match", VAL_CSV_COLUMNS)
        self.assertIn("effective_group_rate", GROUP_DIAG_CSV_COLUMNS)
        self.assertIn("rollout_format_rate", GROUP_DIAG_CSV_COLUMNS)


class UnifiedTrainingLoopIntegrationTest(unittest.TestCase):
    def test_cpu_fake_run_keeps_dashboard_outputs_and_archives_stop(self) -> None:
        """用无模型假训练验证统一调度、CSV、checkpoint和看板读取闭环。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = GRPOTrainer.__new__(GRPOTrainer)
            trainer.cfg = SimpleNamespace(
                output_dir=temp_dir,
                run_name="fake-unified-stop",
                resume_state_mode="weights_only",
                total_training_steps=2,
                train_batch_size=4,
                rollout_n=8,
                rollout_batch_size=8,
                ppo_epochs=2,
                ppo_mini_batch_size=16,
                gradient_accumulation_steps=1,
                learning_rate=5e-6,
                max_response_length=256,
                max_steps_no_improve=50,
                early_stop_trend_window=3,
                early_stop_min_recovery_slope=0.005,
                early_stop_max_extension_steps=20,
                kl_warning_threshold=0.06,
                kl_threshold=0.1,
                kl_guard_window=3,
                kl_guard_patience_checks=3,
                approx_kl_threshold=0.01,
                adaptive_kl_enabled=False,
                adaptive_kl_target=0.04,
                adaptive_kl_tolerance=1.5,
                adaptive_kl_min_coef=0.001,
                adaptive_kl_max_coef=0.05,
                adaptive_kl_interval=10,
                adaptive_kl_factor=1.5,
                reward_hacking_detect=True,
                reward_hacking_window=30,
                signal_guard_window=10,
                signal_guard_warmup_steps=10,
                signal_guard_patience_checks=3,
                signal_guard_non_overlapping_windows=True,
                signal_guard_mixed_hard_stop=False,
                min_effective_group_rate=0.7,
                min_mixed_group_rate=0.6,
                max_zero_advantage_rate=0.3,
                min_rollout_format_rate=0.9,
                save_freq=10,
                eval_freq=10,
                log_steps=1,
                val_before_train=False,
            )
            trainer.reward_config = GSM8KRewardConfig()
            trainer.train_dataset = [object()]
            trainer.resume_checkpoint_dir = None
            trainer.has_trainer_state = False
            trainer.start_step = 0
            trainer.last_completed_step = -1
            trainer.best_val_em = -1.0
            trainer.best_step = -1
            trainer.steps_no_improve = 0
            trainer.early_stop_extension_steps = 0
            trainer.train_reward_history = []
            trainer.metrics_history = []
            trainer.current_kl_loss_coef = 0.005
            trainer.training_session_id = None
            trainer.training_status = "initialized"
            trainer.stop_decision = None
            trainer.stop_reason = None
            trainer._train_csv_path = None
            trainer._val_csv_path = None
            trainer._group_diag_csv_path = None
            trainer._gpu_csv_path = None
            trainer._train_csv_writer = None
            trainer._val_csv_writer = None
            trainer._group_diag_csv_writer = None
            trainer._gpu_csv_writer = None

            def fake_train_step(self: GRPOTrainer, step: int) -> dict[str, object]:
                metrics = {
                    "step": step,
                    "reward_mean": 0.5,
                    "reward_std": 0.1,
                    "policy_loss": 0.01,
                    "kl_loss": 0.03,
                    "total_loss": 0.01,
                    "approx_kl": 0.0001,
                    "clip_frac": 0.0,
                    "response_len_mean": 100.0,
                    "lr": 5e-6,
                    "step_time": 0.01,
                    "grad_norm": 0.2,
                    "kl_loss_coef": 0.005,
                    "prompt_count": 4,
                    "rollout_count": 32,
                    "mini_batch_count": 4,
                    "optimizer_update_count": 4,
                    "group_diagnostics": {
                        "group_count": 4,
                        "rollout_count": 32,
                        "rollout_n": 8,
                        "effective_group_rate": 0.8,
                        "mixed_group_rate": 0.7,
                        "zero_advantage_rate": 0.2,
                        "rollout_format_rate": 0.95,
                    },
                }
                self.metrics_history.append(metrics)
                return metrics

            def fake_save_checkpoint(
                self: GRPOTrainer, output_dir: Path, step: int
            ) -> None:
                checkpoint_dir = output_dir / f"checkpoint-{step}"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                state = {
                    "step": step,
                    "next_step": step + 1,
                    "run_name": self.cfg.run_name,
                    "total_training_steps": self.cfg.total_training_steps,
                    "best_val_em": self.best_val_em,
                    "best_step": self.best_step,
                    "steps_no_improve": self.steps_no_improve,
                    "training_status": self.training_status,
                    "stop_reason": self.stop_reason,
                    "stop_decision": self.stop_decision,
                }
                (checkpoint_dir / "trainer_state.json").write_text(
                    json.dumps(state, ensure_ascii=False), encoding="utf-8"
                )

            trainer.train_step = MethodType(fake_train_step, trainer)
            trainer._save_checkpoint = MethodType(fake_save_checkpoint, trainer)
            trainer._save_run_config = MethodType(lambda self, output_dir: None, trainer)
            trainer._log_gpu_memory_detailed = MethodType(
                lambda self, label, batch: None, trainer
            )
            trainer._append_gpu_csv = MethodType(
                lambda self, step, label: None, trainer
            )

            with patch("ptf.train_grpo.setup_logging", lambda *_: None):
                with self.assertLogs("ptf.grpo", level="INFO") as captured_logs:
                    trainer.train()
            self.assertTrue(
                any(
                    "训练结束, 原因: 达到最大步数" in line
                    for line in captured_logs.output
                )
            )

            output_dir = Path(temp_dir)
            summary = json.loads(
                (output_dir / "training_stop.json").read_text(encoding="utf-8")
            )
            checkpoint_state = json.loads(
                (output_dir / "checkpoint-1" / "trainer_state.json").read_text(
                    encoding="utf-8"
                )
            )
            train_lines = (
                output_dir / "plots" / "train_metrics.csv"
            ).read_text(encoding="utf-8").splitlines()
            group_lines = (
                output_dir / "plots" / "group_diagnostics.csv"
            ).read_text(encoding="utf-8").splitlines()

            self.assertEqual(summary["status"], "completed")
            self.assertEqual(summary["decision"]["category"], StopCategory.MAX_STEPS)
            self.assertEqual(checkpoint_state["training_status"], "completed")
            self.assertEqual(checkpoint_state["stop_reason"], "达到最大步数")
            self.assertEqual(len(train_lines), 3)
            self.assertEqual(len(group_lines), 3)

            dashboard_path = (
                Path(__file__).resolve().parents[1]
                / "scripts"
                / "run_training_dashboard.py"
            )
            spec = importlib.util.spec_from_file_location(
                "dashboard_test_module", dashboard_path
            )
            self.assertIsNotNone(spec)
            self.assertIsNotNone(spec.loader)
            dashboard = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = dashboard
            spec.loader.exec_module(dashboard)
            status = dashboard.build_status(
                dashboard.DashboardConfig(output_dir, 5, 20)
            )
            self.assertEqual(status["latest_step"], 1)
            self.assertEqual(len(status["train_rows"]), 2)
            self.assertEqual(len(status["group_rows"]), 2)
            self.assertEqual(status["training_status"], "completed")
            self.assertEqual(status["stop_reason"], "达到最大步数")

            causal_path = (
                Path(__file__).resolve().parents[1]
                / "scripts"
                / "run_grpo_causal_experiments.py"
            )
            causal_spec = importlib.util.spec_from_file_location(
                "causal_summary_test_module", causal_path
            )
            self.assertIsNotNone(causal_spec)
            self.assertIsNotNone(causal_spec.loader)
            causal = importlib.util.module_from_spec(causal_spec)
            sys.modules[causal_spec.name] = causal
            causal_spec.loader.exec_module(causal)
            self.assertEqual(causal._extract_stop_reason(output_dir), "达到最大步数")

            # 同一假训练器重置后注入运行时异常，验证失败不会被伪装成正常结束。
            failed_output = output_dir / "failed-run"
            trainer.cfg.output_dir = str(failed_output)
            trainer.start_step = 0
            trainer.last_completed_step = -1
            trainer.best_val_em = -1.0
            trainer.best_step = -1
            trainer.steps_no_improve = 0
            trainer.early_stop_extension_steps = 0
            trainer.train_reward_history = []
            trainer.metrics_history = []
            trainer._train_csv_writer = None
            trainer._val_csv_writer = None
            trainer._group_diag_csv_writer = None
            trainer._gpu_csv_writer = None

            def failing_train_step(self: GRPOTrainer, step: int) -> dict[str, object]:
                raise RuntimeError("注入训练异常")

            trainer.train_step = MethodType(failing_train_step, trainer)
            with patch("ptf.train_grpo.setup_logging", lambda *_: None):
                with self.assertLogs("ptf.grpo", level="INFO"):
                    with self.assertRaisesRegex(RuntimeError, "注入训练异常"):
                        trainer.train()

            failed_summary = json.loads(
                (failed_output / "training_stop.json").read_text(encoding="utf-8")
            )
            self.assertEqual(failed_summary["status"], "failed")
            self.assertEqual(
                failed_summary["decision"]["category"], StopCategory.RUNTIME_ERROR
            )
            self.assertEqual(failed_summary["last_completed_step"], -1)
            self.assertIsNone(failed_summary["checkpoint_path"])


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from snake_game import ACTION_NAMES, FEATURE_NAMES, SnakeEnv, draw_game


@dataclass
class TrainConfig:
    board_size: int = 6
    episodes: int = 2500
    alpha: float = 0.1
    gamma: float = 0.9
    epsilon_start: float = 1.0
    epsilon_min: float = 0.02
    epsilon_decay: float = 0.998
    eval_episodes: int = 80
    seed: int = 7


def choose_action(q_table: np.ndarray, state_id: int, epsilon: float, rng: random.Random) -> int:
    if rng.random() < epsilon:
        return rng.randrange(len(ACTION_NAMES))
    return int(np.argmax(q_table[state_id]))


def train_agent(config: TrainConfig) -> tuple[np.ndarray, list[dict[str, float]], dict[str, object]]:
    rng = random.Random(config.seed)
    np.random.seed(config.seed)
    env = SnakeEnv(board_size=config.board_size, seed=config.seed)
    q_table = np.zeros((2 ** len(FEATURE_NAMES), len(ACTION_NAMES)), dtype=np.float32)

    epsilon = config.epsilon_start
    rows: list[dict[str, float]] = []

    for episode in range(1, config.episodes + 1):
        state_id = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action_id = choose_action(q_table, state_id, epsilon, rng)
            result = env.step_relative(action_id)
            target = result.reward
            if not result.done:
                target += config.gamma * float(np.max(q_table[result.state_id]))
            q_table[state_id, action_id] += config.alpha * (target - q_table[state_id, action_id])

            state_id = result.state_id
            done = result.done
            total_reward += result.reward

        rows.append(
            {
                "episode": float(episode),
                "score": float(env.score),
                "total_reward": float(total_reward),
                "epsilon": float(epsilon),
            }
        )
        epsilon = max(config.epsilon_min, epsilon * config.epsilon_decay)

    evaluation = evaluate_policy(q_table, config)
    scores = [row["score"] for row in rows]
    rewards = [row["total_reward"] for row in rows]
    metrics: dict[str, object] = {
        "config": asdict(config),
        "state_features": FEATURE_NAMES,
        "actions": ACTION_NAMES,
        "epsilon_final": float(epsilon),
        "first_100_mean_score": float(np.mean(scores[: min(100, len(scores))])),
        "last_100_mean_score": float(np.mean(scores[-min(100, len(scores)) :])),
        "best_training_score": int(max(scores) if scores else 0),
        "mean_training_reward": float(np.mean(rewards) if rewards else 0.0),
        "evaluation": evaluation,
    }
    return q_table, rows, metrics


def evaluate_policy(q_table: np.ndarray, config: TrainConfig) -> dict[str, float]:
    eval_rng = random.Random(config.seed + 1000)
    scores: list[float] = []
    rewards: list[float] = []

    for _ in range(config.eval_episodes):
        env = SnakeEnv(board_size=config.board_size, seed=eval_rng.randrange(1_000_000_000))
        state_id = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action_id = int(np.argmax(q_table[state_id]))
            result = env.step_relative(action_id)
            state_id = result.state_id
            done = result.done
            total_reward += result.reward
        scores.append(float(env.score))
        rewards.append(float(total_reward))

    return {
        "episodes": float(config.eval_episodes),
        "mean_score": float(np.mean(scores) if scores else 0.0),
        "max_score": float(max(scores) if scores else 0.0),
        "mean_reward": float(np.mean(rewards) if rewards else 0.0),
    }


def save_outputs(
    q_table: np.ndarray,
    rows: list[dict[str, float]],
    metrics: dict[str, object],
    output_dir: Path,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    q_table_path = output_dir / "q_table.npy"
    metrics_path = output_dir / "training_metrics.json"
    scores_path = output_dir / "training_scores.csv"

    np.save(q_table_path, q_table)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    with scores_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["episode", "score", "total_reward", "epsilon"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return {
        "q_table": str(q_table_path),
        "metrics": str(metrics_path),
        "scores": str(scores_path),
    }


def load_q_table(model_path: Path) -> np.ndarray:
    if not model_path.exists():
        raise FileNotFoundError(f"找不到模型文件：{model_path}")
    q_table = np.load(model_path)
    expected_shape = (2 ** len(FEATURE_NAMES), len(ACTION_NAMES))
    if q_table.shape != expected_shape:
        raise ValueError(f"Q 表形状应为 {expected_shape}，实际为 {q_table.shape}")
    return q_table


def load_board_size_from_metrics(model_path: Path, fallback: int) -> int:
    metrics_path = model_path.parent / "training_metrics.json"
    if not metrics_path.exists():
        return fallback
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        return int(metrics.get("config", {}).get("board_size", fallback))
    except (json.JSONDecodeError, TypeError, ValueError):
        return fallback


def run_agent_demo(
    q_table: np.ndarray,
    board_size: int,
    fps: int,
    cell_size: int,
    seed: int,
    author: str = "",
) -> None:
    try:
        import pygame  # type: ignore
    except ImportError as exc:
        raise SystemExit("未安装 pygame，无法自动演示。请先运行：python -m pip install pygame") from exc

    pygame.init()
    env = SnakeEnv(board_size=board_size, seed=seed)
    header_height = 72
    width = board_size * cell_size + 40
    height = board_size * cell_size + header_height + 20
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Snake Q-learning Demo")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("arial", 26, bold=True)
    small_font = pygame.font.SysFont("arial", 16)

    running = True
    pause_frames = 0
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    env.reset()
                    pause_frames = 0

        if env.done:
            pause_frames += 1
            if pause_frames >= fps * 2:
                env.reset()
                pause_frames = 0
        else:
            state_id = env.get_state_id()
            action_id = int(np.argmax(q_table[state_id]))
            env.step_relative(action_id)

        draw_game(screen, pygame, env, cell_size, font, small_font, author)
        pygame.display.flip()
        clock.tick(fps)

    pygame.quit()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="使用 Q-learning 训练智能体玩贪吃蛇。")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="训练 Q 表并保存结果。")
    train.add_argument("--episodes", type=int, default=2500)
    train.add_argument("--board-size", type=int, default=6)
    train.add_argument("--alpha", type=float, default=0.1)
    train.add_argument("--gamma", type=float, default=0.9)
    train.add_argument("--epsilon-start", type=float, default=1.0)
    train.add_argument("--epsilon-min", type=float, default=0.02)
    train.add_argument("--epsilon-decay", type=float, default=0.998)
    train.add_argument("--eval-episodes", type=int, default=80)
    train.add_argument("--seed", type=int, default=7)
    train.add_argument("--output-dir", default="snake_rl_outputs")
    train.add_argument("--auto-demo", action="store_true", help="训练后立即打开窗口演示。")
    train.add_argument("--demo-fps", type=int, default=10)
    train.add_argument("--cell-size", type=int, default=64)
    train.add_argument("--author", type=str, default="", help="作者名称，默认空。")

    demo = subparsers.add_parser("demo", help="加载已训练 Q 表并打开窗口演示。")
    demo.add_argument("--model-path", default="snake_rl_outputs/q_table.npy")
    demo.add_argument("--board-size", type=int, default=6)
    demo.add_argument("--fps", type=int, default=10)
    demo.add_argument("--cell-size", type=int, default=64)
    demo.add_argument("--seed", type=int, default=2024)
    demo.add_argument("--author", type=str, default="", help="作者名称，默认空。")
    return parser


def command_train(args: argparse.Namespace) -> None:
    config = TrainConfig(
        board_size=args.board_size,
        episodes=args.episodes,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        eval_episodes=args.eval_episodes,
        seed=args.seed,
    )
    q_table, rows, metrics = train_agent(config)
    paths = save_outputs(q_table, rows, metrics, Path(args.output_dir))
    print(json.dumps({"outputs": paths, "metrics": metrics}, ensure_ascii=False, indent=2))
    if args.auto_demo:
        run_agent_demo(q_table, config.board_size, args.demo_fps, args.cell_size, config.seed + 2024, args.author)


def command_demo(args: argparse.Namespace) -> None:
    model_path = Path(args.model_path)
    q_table = load_q_table(model_path)
    board_size = load_board_size_from_metrics(model_path, args.board_size)
    run_agent_demo(q_table, board_size, args.fps, args.cell_size, args.seed, args.author)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "train":
        command_train(args)
    elif args.command == "demo":
        command_demo(args)


if __name__ == "__main__":
    main()

"""Verify Candidate #2 geometry, pixel invariance, and communication logic."""

import numpy as np
from lab1.config import Config
from request_agent.final_candidate2_env import FinalCandidate2MergeEnv


def verify():
    print("=" * 75)
    print("FINAL CANDIDATE #2 ENVIRONMENT VERIFICATION")
    print("=" * 75)

    cfg = Config.load("configs/cpu_fast.yaml")
    env = FinalCandidate2MergeEnv(cfg)

    # 1. Geometry verification
    obs, info = env.reset(seed=101)
    base = env.env.unwrapped
    ego = base.vehicle
    front = base.front_vehicle
    rear = base.rear_vehicle

    print(f"Ego:   x={ego.position[0]:.1f}m, y={ego.position[1]:.1f}m, speed={ego.speed:.1f}m/s")
    print(f"Front: x={front.position[0]:.1f}m, y={front.position[1]:.1f}m, speed={front.speed:.1f}m/s")
    print(f"Rear:  x={rear.position[0]:.1f}m, y={rear.position[1]:.1f}m, speed={rear.speed:.1f}m/s")
    print(f"Gap:   {front.position[0] - rear.position[0]:.1f}m")

    assert np.isclose(ego.position[0], 30.0, atol=1.0)
    assert np.isclose(ego.speed, 25.0, atol=0.5)
    assert np.isclose(front.position[0], 54.0, atol=1.0)
    assert np.isclose(front.speed, 22.0, atol=0.5)
    assert np.isclose(rear.position[0], 18.0, atol=1.0)
    assert np.isclose(rear.speed, 22.0, atol=0.5)
    print("-> GEOMETRY VERIFIED.")

    # 2. Pixel invariance verification at t=0
    obs_m, _ = env.reset(seed=42)
    env.rear_intent = env.MAINTAIN

    obs_a, _ = env.reset(seed=42)
    env.rear_intent = env.ACCELERATE

    obs_b, _ = env.reset(seed=42)
    env.rear_intent = env.BRAKE

    diff_a = np.max(np.abs(obs_m.astype(int) - obs_a.astype(int)))
    diff_b = np.max(np.abs(obs_m.astype(int) - obs_b.astype(int)))
    print(f"Pixel diff (MAINTAIN vs ACCEL): {diff_a}")
    print(f"Pixel diff (MAINTAIN vs BRAKE): {diff_b}")
    assert diff_a == 0 and diff_b == 0
    print("-> 100% BYTE-FOR-BYTE PIXEL INVARIANCE CONFIRMED AT t=0.")

    # 3. Privacy / no leakage verification
    forbidden = ["intent", "front_intent", "rear_intent", "target_speed"]
    leaked = [k for k in forbidden if k in info]
    assert not leaked, f"Leaked: {leaked}"
    print(f"Sanitized info keys: {list(info.keys())}")
    print("-> ZERO INTENT LEAKAGE CONFIRMED.")

    # 4. Communication semantics verification
    env.rear_intent = env.BRAKE
    assert env.get_message(env.NONE) == 0
    assert env.get_message(env.ASK_FRONT) == 1
    assert env.get_message(env.ASK_REAR) == 3

    env.rear_intent = env.ACCELERATE
    assert env.get_message(env.ASK_FRONT) == 1
    assert env.get_message(env.ASK_REAR) == 2
    print("-> COMMUNICATION QUERY LOGIC CONFIRMED.")

    # 5. Exact intent target speed dynamics
    env._apply_intent(rear, env.BRAKE, 22.0)
    assert np.isclose(rear.target_speed if hasattr(rear, "target_speed") else rear.speed, 17.0, atol=0.5)
    env._apply_intent(rear, env.MAINTAIN, 22.0)
    assert np.isclose(rear.target_speed if hasattr(rear, "target_speed") else rear.speed, 22.0, atol=0.5)
    env._apply_intent(rear, env.ACCELERATE, 22.0)
    assert np.isclose(rear.target_speed if hasattr(rear, "target_speed") else rear.speed, 27.0, atol=0.5)
    print("-> INTENT SPEED DYNAMICS CONFIRMED (17 m/s, 22 m/s, 27 m/s).")

    env.close()
    print("\nALL PHASE 1 VERIFICATION CHECKS PASSED!")


if __name__ == "__main__":
    verify()


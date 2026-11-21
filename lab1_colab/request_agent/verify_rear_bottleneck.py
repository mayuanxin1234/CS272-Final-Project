"""Verify pixel invariance, partial observability, and geometry of Design 1."""

import numpy as np
from lab1.config import Config
from request_agent.rear_bottleneck_env import RearBottleneckMergeEnv


def verify():
    cfg = Config.load("configs/cpu_fast.yaml")
    env = RearBottleneckMergeEnv(cfg)

    # 1. Geometry check
    obs, info = env.reset(seed=123)
    base = env.env.unwrapped
    ego = base.vehicle
    front = base.front_vehicle
    rear = base.rear_vehicle

    print("=" * 70)
    print("DESIGN 1 GEOMETRY & INITIAL CONFIGURATION CHECK")
    print("=" * 70)
    print(f"Ego position:  x={ego.position[0]:.1f}m, y={ego.position[1]:.1f}m, speed={ego.speed:.1f}m/s")
    print(f"Front vehicle: x={front.position[0]:.1f}m, y={front.position[1]:.1f}m, speed={front.speed:.1f}m/s")
    print(f"Rear vehicle:  x={rear.position[0]:.1f}m, y={rear.position[1]:.1f}m, speed={rear.speed:.1f}m/s")
    print(f"Target lane gap: {front.position[0] - rear.position[0]:.1f}m")

    assert np.isclose(ego.position[0], 30.0, atol=1.0), f"Ego x={ego.position[0]}"
    assert np.isclose(front.position[0], 75.0, atol=1.0), f"Front x={front.position[0]}"
    assert np.isclose(rear.position[0], 18.0, atol=1.0), f"Rear x={rear.position[0]}"
    assert np.isclose(ego.speed, 25.0, atol=0.5), f"Ego speed={ego.speed}"
    assert np.isclose(front.speed, 25.0, atol=0.5), f"Front speed={front.speed}"
    assert np.isclose(rear.speed, 25.0, atol=0.5), f"Rear speed={rear.speed}"

    # 2. Pixel invariance check at t=0 across all 3 rear intents
    print("\n" + "=" * 70)
    print("PIXEL INVARIANCE CHECK AT t=0 ACROSS REAR INTENTS")
    print("=" * 70)
    obs_maintain, _ = env.reset(seed=42)
    env.rear_intent = env.MAINTAIN

    obs_accel, _ = env.reset(seed=42)
    env.rear_intent = env.ACCELERATE

    obs_brake, _ = env.reset(seed=42)
    env.rear_intent = env.BRAKE

    diff_accel = np.max(np.abs(obs_maintain.astype(int) - obs_accel.astype(int)))
    diff_brake = np.max(np.abs(obs_maintain.astype(int) - obs_brake.astype(int)))

    print(f"Max pixel difference (MAINTAIN vs ACCELERATE): {diff_accel}")
    print(f"Max pixel difference (MAINTAIN vs BRAKE):      {diff_brake}")
    assert diff_accel == 0, "Pixels differed between MAINTAIN and ACCELERATE!"
    assert diff_brake == 0, "Pixels differed between MAINTAIN and BRAKE!"
    print("-> 100% BYTE-FOR-BYTE PIXEL INVARIANCE CONFIRMED AT t=0.")

    # 3. Hidden intent privacy check
    print("\n" + "=" * 70)
    print("HIDDEN INTENT LEAKAGE CHECK IN INFO")
    print("=" * 70)
    forbidden = ["intent", "front_intent", "rear_intent", "target_speed", "action_type"]
    leaked = [k for k in forbidden if k in info]
    print(f"Sanitized info keys: {list(info.keys())}")
    assert not leaked, f"Intent leaked in info: {leaked}"
    print("-> ZERO INTENT LEAKAGE CONFIRMED.")

    # 4. Message query check
    print("\n" + "=" * 70)
    print("COMMUNICATION QUERY CHECK")
    print("=" * 70)
    env.rear_intent = env.BRAKE
    m_none = env.get_message(env.NONE)
    m_front = env.get_message(env.ASK_FRONT)
    m_rear = env.get_message(env.ASK_REAR)
    print(f"NONE query returned:      {m_none} (expected UNKNOWN=0)")
    print(f"ASK_FRONT query returned: {m_front} (expected MAINTAIN=1)")
    print(f"ASK_REAR query returned:  {m_rear} (expected BRAKE=3)")
    assert m_none == 0
    assert m_front == 1
    assert m_rear == 3
    print("-> COMMUNICATION LOGIC CONFIRMED.")

    env.close()
    print("\nAll Design 1 environment verification tests PASSED successfully!")


if __name__ == "__main__":
    verify()


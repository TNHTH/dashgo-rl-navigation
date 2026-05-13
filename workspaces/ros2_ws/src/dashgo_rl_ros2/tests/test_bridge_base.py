from __future__ import annotations

from dataclasses import dataclass

from dashgo_rl_ros2.bridge_base import BridgeCommandPublisher, DiagnosticStatusBuilder, is_stale


@dataclass
class FakeTwist:
    linear_x: float = 0.0
    angular_z: float = 0.0


class FakePublisher:
    def __init__(self) -> None:
        self.messages = []

    def publish(self, msg) -> None:
        self.messages.append(msg)


class FakeDiagnosticStatus:
    OK = 0
    WARN = 1
    ERROR = 2

    def __init__(self) -> None:
        self.name = ""
        self.hardware_id = ""
        self.level = self.OK
        self.message = ""
        self.values = []


@dataclass
class FakeKeyValue:
    key: str = ""
    value: str = ""


def test_is_stale_treats_missing_stamp_and_positive_timeout_as_stale() -> None:
    assert is_stale(now_sec=10.0, stamp_sec=None, timeout_sec=1.0)
    assert is_stale(now_sec=10.0, stamp_sec=8.0, timeout_sec=1.0)
    assert not is_stale(now_sec=10.0, stamp_sec=9.5, timeout_sec=1.0)
    assert not is_stale(now_sec=10.0, stamp_sec=1.0, timeout_sec=0.0)


def test_bridge_command_publisher_respects_shadow_mode() -> None:
    cmd_pub = FakePublisher()
    debug_pub = FakePublisher()
    publisher = BridgeCommandPublisher(cmd_pub=cmd_pub, debug_cmd_pub=debug_pub, shadow_mode=True)
    twist = FakeTwist(linear_x=0.1, angular_z=0.2)

    publisher.publish(twist)

    assert debug_pub.messages == [twist]
    assert cmd_pub.messages == []


def test_bridge_command_publisher_publishes_to_cmd_when_not_shadowed() -> None:
    cmd_pub = FakePublisher()
    debug_pub = FakePublisher()
    publisher = BridgeCommandPublisher(cmd_pub=cmd_pub, debug_cmd_pub=debug_pub, shadow_mode=False)
    twist = FakeTwist(linear_x=0.1, angular_z=0.2)

    publisher.publish(twist)

    assert debug_pub.messages == [twist]
    assert cmd_pub.messages == [twist]


def test_diagnostic_status_builder_formats_values() -> None:
    builder = DiagnosticStatusBuilder(
        diagnostic_status_cls=FakeDiagnosticStatus,
        key_value_cls=FakeKeyValue,
    )

    status = builder.build(
        name="node",
        hardware_id="hardware",
        level=FakeDiagnosticStatus.WARN,
        message="waiting",
        values={
            "ready": True,
            "age": None,
            "distance": "1.250",
        },
    )

    assert status.name == "node"
    assert status.hardware_id == "hardware"
    assert status.level == FakeDiagnosticStatus.WARN
    assert status.message == "waiting"
    assert [(item.key, item.value) for item in status.values] == [
        ("ready", "true"),
        ("age", ""),
        ("distance", "1.250"),
    ]

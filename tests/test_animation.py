"""Tests for the animation engine."""

import time

from tpet.animation.engine import AnimationState, PetAnimator


class TestAnimationState:
    def test_values(self) -> None:
        assert AnimationState.IDLE.value == "idle"
        assert AnimationState.REACTING.value == "reacting"
        assert AnimationState.SLEEPING.value == "sleeping"


class TestPetAnimator:
    def test_initial_state(self) -> None:
        animator = PetAnimator(frame_count=4, idle_duration=3.0, reaction_duration=0.5, sleep_threshold=120)
        assert animator.state == AnimationState.IDLE
        assert animator.current_frame == 0

    def test_idle_frame_index(self) -> None:
        animator = PetAnimator(frame_count=4, idle_duration=0.01, reaction_duration=0.5, sleep_threshold=120)
        assert animator.current_frame in (0, 1)

    def test_react_changes_state(self) -> None:
        animator = PetAnimator(frame_count=4, idle_duration=3.0, reaction_duration=0.5, sleep_threshold=120)
        animator.react()
        assert animator.state == AnimationState.REACTING
        assert animator.current_frame == 2

    def test_reaction_expires(self) -> None:
        animator = PetAnimator(frame_count=4, idle_duration=3.0, reaction_duration=0.01, sleep_threshold=120)
        animator.react()
        time.sleep(0.05)
        animator.tick()
        assert animator.state == AnimationState.IDLE

    def test_sleep_after_threshold(self) -> None:
        animator = PetAnimator(frame_count=4, idle_duration=3.0, reaction_duration=0.5, sleep_threshold=0)
        time.sleep(0.01)
        animator.tick()
        assert animator.state == AnimationState.SLEEPING
        assert animator.current_frame == 3

    def test_wake_on_activity(self) -> None:
        animator = PetAnimator(frame_count=4, idle_duration=3.0, reaction_duration=0.5, sleep_threshold=0)
        time.sleep(0.01)
        animator.tick()
        assert animator.state == AnimationState.SLEEPING
        animator.react()
        assert animator.state == AnimationState.REACTING

    def test_idle_frame_cycling(self) -> None:
        animator = PetAnimator(frame_count=4, idle_duration=0.01, reaction_duration=0.5, sleep_threshold=9999)
        frames_seen: set[int] = set()
        for _ in range(20):
            animator.tick()
            frames_seen.add(animator.current_frame)
            time.sleep(0.01)
        assert 0 in frames_seen
        assert 1 in frames_seen

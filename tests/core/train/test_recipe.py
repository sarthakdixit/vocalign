from core.train import recipe


def test_below_hard_minimum_is_zero_shot_only():
    r = recipe.select_recipe(10.0)

    assert r.tier == "zero_shot_only"
    assert r.attempt_training is False
    assert r.gpt_epochs == 0


def test_each_tier_boundary_is_exclusive_on_the_lower_tier():
    cases = [
        (recipe.ZERO_SHOT_ONLY_BELOW_SECONDS - 1, "zero_shot_only"),
        (recipe.ZERO_SHOT_ONLY_BELOW_SECONDS, "minimal"),
        (recipe.MINIMAL_BELOW_SECONDS - 1, "minimal"),
        (recipe.MINIMAL_BELOW_SECONDS, "light"),
        (recipe.LIGHT_BELOW_SECONDS - 1, "light"),
        (recipe.LIGHT_BELOW_SECONDS, "standard"),
        (recipe.STANDARD_BELOW_SECONDS - 1, "standard"),
        (recipe.STANDARD_BELOW_SECONDS, "extended"),
    ]
    for seconds, expected_tier in cases:
        assert recipe.select_recipe(seconds).tier == expected_tier, seconds


def test_epoch_counts_increase_with_tier():
    epochs = [recipe.select_recipe(s).gpt_epochs for s in (30.0, 120.0, 600.0, 3000.0)]

    assert epochs == sorted(epochs)
    assert epochs[0] < epochs[-1]


def test_attempt_training_is_false_only_for_zero_shot_tier():
    for seconds in (30.0, 120.0, 600.0, 3000.0):
        assert recipe.select_recipe(seconds).attempt_training is True
    assert recipe.select_recipe(5.0).attempt_training is False

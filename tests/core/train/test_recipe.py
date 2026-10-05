from core.train import recipe


def test_below_hard_minimum_is_zero_shot_only():
    r = recipe.select_recipe(10.0)

    assert r.tier == "zero_shot_only"
    assert r.attempt_training is False
    assert r.gpt_epochs == 0 and r.sovits_epochs == 0
    assert r.lora_rank is None


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


def test_v2_family_never_gets_lora_rank_at_any_tier():
    for version in ("v2", "v2Pro", "v2ProPlus"):
        for seconds in (30.0, 120.0, 600.0, 3000.0):
            assert recipe.select_recipe(seconds, version=version).lora_rank is None


def test_v3_and_v4_get_lora_rank_whenever_training_is_attempted():
    for version in ("v3", "v4"):
        for seconds in (30.0, 120.0, 600.0, 3000.0):
            assert recipe.select_recipe(seconds, version=version).lora_rank is not None


def test_zero_shot_tier_has_no_lora_rank_even_for_lora_capable_versions():
    r = recipe.select_recipe(5.0, version="v3")

    assert r.attempt_training is False
    assert r.lora_rank is None


def test_lora_rank_increases_with_tier_for_v3():
    ranks = [recipe.select_recipe(s, version="v3").lora_rank for s in (30.0, 120.0, 600.0, 3000.0)]

    assert ranks == sorted(ranks)
    assert ranks[0] < ranks[-1]


def test_epoch_counts_increase_with_tier():
    epochs = [recipe.select_recipe(s).gpt_epochs for s in (30.0, 120.0, 600.0, 3000.0)]

    assert epochs == sorted(epochs)
    assert epochs[0] < epochs[-1]


def test_default_version_gets_no_lora_rank():
    assert recipe.select_recipe(120.0).lora_rank is None

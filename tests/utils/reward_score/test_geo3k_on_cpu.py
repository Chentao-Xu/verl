# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest

from verl.utils.reward_score import default_compute_score
from verl.utils.reward_score.geo3k import compute_score


@pytest.mark.parametrize(
    ("solution", "expected_score"),
    [
        (r"<think>The two equidistant chords are congruent.</think> Therefore, \boxed{48}.", 1.0),
        (r"<think>The two equidistant chords are congruent.</think> Therefore, \boxed{47}.", 0.1),
        (r"The chord length is \boxed{48}.", 0.9),
        (r"The chord length is \boxed{47}.", 0.0),
    ],
)
def test_geo3k_reward_distinguishes_accuracy_and_format(solution, expected_score):
    assert compute_score(solution, "48") == pytest.approx(expected_score)


def test_default_reward_router_uses_geo3k_score():
    solution = r"<think>The two equidistant chords are congruent.</think> Therefore, \boxed{48}."

    score = default_compute_score(
        data_source="hiyouga/geometry3k",
        solution_str=solution,
        ground_truth="48",
    )

    assert score == pytest.approx(1.0)

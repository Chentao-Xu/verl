# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

from verl.utils.reward_score import default_compute_score
from verl.utils.reward_score.virl39k import accuracy_reward, compute_score, format_reward


def test_boxed_correct_answer_gets_full_reward():
    assert compute_score(r"Reasoning. Final answer: \boxed{A}", r"\boxed{A}") == 1.0


def test_correct_answer_without_box_loses_only_format_reward():
    assert compute_score("A", r"\boxed{A}") == 0.9


def test_wrong_boxed_answer_gets_only_format_reward():
    assert compute_score(r"\boxed{B}", r"\boxed{A}") == 0.1


def test_mathruler_normalizes_equivalent_answers():
    assert accuracy_reward(r"\boxed{1/2}", r"\boxed{\frac{1}{2}}") == 1.0
    assert accuracy_reward(r"\boxed{17}", r"\boxed{17^\circ}") == 1.0


def test_format_reward_requires_boxed_output():
    assert format_reward(r"\boxed{42}") == 1.0
    assert format_reward("42") == 0.0


def test_default_dispatcher_supports_virl39k():
    assert (
        default_compute_score(
            data_source="TIGER-Lab/ViRL39K",
            solution_str=r"\boxed{42}",
            ground_truth=r"\boxed{42}",
        )
        == 1.0
    )

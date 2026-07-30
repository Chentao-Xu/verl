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
"""Rule-based reward for TIGER-Lab/ViRL39K."""

import re

from mathruler.grader import extract_boxed_content, grade_answer


def _extract_answer(text: str) -> str:
    boxed = extract_boxed_content(text)
    return boxed if boxed and boxed != "None" else text.strip()


def format_reward(predict_str: str) -> float:
    return 1.0 if re.search(r"\\boxed\s*\{.+\}", predict_str, re.DOTALL) else 0.0


def accuracy_reward(predict_str: str, ground_truth: str) -> float:
    prediction = _extract_answer(predict_str)
    target = _extract_answer(ground_truth)
    return 1.0 if grade_answer(prediction, target) else 0.0


def compute_score(predict_str: str, ground_truth: str, format_score: float = 0.1) -> float:
    """Return correctness plus a small reward for the required boxed format."""
    accuracy = accuracy_reward(predict_str, ground_truth)
    return (1.0 - format_score) * accuracy + format_score * format_reward(predict_str)

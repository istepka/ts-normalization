#  Copyright (c) 2024, Salesforce, Inc.
#  SPDX-License-Identifier: Apache-2
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import torch


def packed_attention_mask(sample_id: torch.Tensor) -> torch.Tensor:
    sample_id = sample_id.unsqueeze(-1)
    attention_mask = sample_id.eq(sample_id.mT)
    return attention_mask


def packed_causal_attention_mask(
    sample_id: torch.Tensor,
    time_id: torch.Tensor,
) -> torch.Tensor:
    attention_mask = packed_attention_mask(sample_id)
    expanded_id1 = time_id.unsqueeze(-2)
    expanded_id2 = time_id.unsqueeze(-1)
    compare_res = expanded_id1 <= expanded_id2
    attention_mask = attention_mask * compare_res
    return attention_mask


def safe_div(
    numer: torch.Tensor,
    denom: torch.Tensor,
) -> torch.Tensor:
    return numer / torch.where(
        denom == 0,
        1.0,
        denom,
    )

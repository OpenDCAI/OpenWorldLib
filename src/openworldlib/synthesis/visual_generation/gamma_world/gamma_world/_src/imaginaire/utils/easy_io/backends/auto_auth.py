# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import contextlib
import json
import os
from collections.abc import Generator
from typing import IO, Any, Optional, Union

@contextlib.contextmanager
def open_auth(s3_credential_path: Optional[Any], mode: str) -> Generator[Union[None, dict[str, Any], IO]]:
    if not s3_credential_path:
        yield None
        return
    if not os.path.exists(s3_credential_path):
        raise FileNotFoundError(f"S3 credential file not found: {s3_credential_path}")
    with open(s3_credential_path, mode) as f:
        yield f


def json_load_auth(f: Union[None, dict[str, Any], IO]) -> dict[str, Any]:
    # None.
    if f is None:
        return {}
    # dict[str, Any].
    elif isinstance(f, dict):
        return f
    # IO.
    else:
        return json.load(f)

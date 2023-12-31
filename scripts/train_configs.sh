#!/bin/bash
# Copyright 2019 Google LLC
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
# ==============================================================================
# This scripts performs local training for a PyTorch model.
echo "Training local ML model"

python train.py --config=configs/GNN_HopperBulletEnv-v0.yaml
python train.py --config=configs/GNN_InvertedDoublePendulumBulletEnv-v0.yaml
python train.py --config=configs/GNN_InvertedPendulumBulletEnv-v0.yaml
python train.py --config=configs/GNN_ReacherBulletEnv-v0.yaml
python train.py --config=configs/GNN_Walker2DBulletEnv-v0.yaml

##############
# Filename:  mvcg.py
# Author:    Sungsoo Kim (sungsoo@etri.re.kr)
# Date:      2026년 3월 10일
# Description: 하이퍼 관계형 지식 그래프(HRKG)를 활용한 보이스피싱 탐지 모델을 구현하는 메인 스크립트
#              이 리팩토링된 버전은 --gnn_type이 지정되지 않은 경우 모든 GNN 모델에 대해 학습, 평가를 자동 수행하고
#              그 결과를 텍스트 파일과 성능 비교 그래프로 저장합니다. 전역 변수 의존성을 제거하여 안정성을 향상시켰으며,
#              accuracy_score 함수에 존재하지 않는 'zero_division' 인자를 제거하여 Type Error를 해결했다.
#              새로운 CMVHRKG (Contrastive Multi-View Hyper-Relational Knowledge Graph) 모델을 추가했다.
#
# Copyright © 2026, Electronics and Telecommunications Research Institute. All Rights Reserved.
##############
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, HeteroData
from torch_geometric.nn import RGCNConv, SAGEConv, HGTConv, HANConv, HeteroConv, FiLMConv, GCNConv, GINConv
import numpy as np
import pandas as pd
import os
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import sys
import json
import glob
from alive_progress import alive_bar
from abc import ABC, abstractmethod
from typing import Union, Tuple, Optional, Dict, List
import warnings
from torch_geometric.utils import to_undirected
import shutil

# PyTorch 2.6+에서 torch.load 보안 문제 해결을 위한 import
import torch.serialization
from torch_geometric.data.storage import BaseStorage

# 터미널 출력을 위한 ANSI 색상 및 스타일 클래스
class AnsiColors:
    """터미널 출력을 위한 ANSI 색상 및 스타일 클래스"""
    RESET = '\033[0m'
    BOLD = '\033[1m'
    CYAN = '\033[36m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    RED = '\033[31m'
    
def print_banner():
    """
    프로그램의 배너를 터미널에 출력합니다.
    """
    try:
        columns = shutil.get_terminal_size().columns
    except OSError:
        columns = 80

    program_name = "ETRI's Voice Phishing Detector using Hyper-Relational Knowledge Graphs"
    version = "v1.0.0"
    copyright_info = "Copyright © 2026, Electronics and Telecommunications Research Institute. All Rights Reserved."
    developer_info = "Developed by: Sungsoo Kim (sungsoo@etri.re.kr)"

    banner_lines: List[str] = [
        f"{AnsiColors.CYAN}{AnsiColors.BOLD}{program_name}{AnsiColors.RESET}",
        f"{AnsiColors.GREEN}Version: {version}{AnsiColors.RESET}",
        f"{AnsiColors.YELLOW}{copyright_info}{AnsiColors.RESET}",
        f"{AnsiColors.GREEN}{developer_info}{AnsiColors.RESET}"
    ]

    separator = "=" * columns

    print(f"\n{AnsiColors.CYAN}{separator}{AnsiColors.RESET}\n")
    for line in banner_lines:
        print(line.center(columns))
    print(f"\n{AnsiColors.CYAN}{separator}{AnsiColors.RESET}\n")

print_banner()

# ==============================================================================
# GPU 설정 및 초기화
# ==============================================================================
def get_most_available_gpu():
    """
    다중 GPU 환경에서 현재 사용 가능한 메모리가 가장 많은 GPU를 찾습니다.
    """
    if not torch.cuda.is_available():
        return torch.device('cpu')
    
    num_gpus = torch.cuda.device_count()
    if num_gpus == 1:
        return torch.device('cuda:0')

    max_free_memory = -1
    best_gpu_index = 0
    
    print(f"시스템에 총 {num_gpus}개의 GPU가 탐지되었습니다. 가장 여유로운 GPU를 선택합니다.")

    for i in range(num_gpus):
        try:
            free_memory, total_memory = torch.cuda.mem_get_info(i)
            print(f"  - GPU {i}: 사용 가능 메모리: {free_memory / 1024**3:.2f} GB / 총 메모리: {total_memory / 1024**3:.2f} GB")
            if free_memory > max_free_memory:
                max_free_memory = free_memory
                best_gpu_index = i
        except Exception as e:
            print(f"Warning: Failed to get memory info for GPU {i}. Error: {e}", file=sys.stderr)
            
    return torch.device(f'cuda:{best_gpu_index}')

device = get_most_available_gpu()
print(f"현재 연산 장치: {device}")

try:
    from transformers import AutoTokenizer, AutoModel
    MODEL_NAME = "monologg/kobert"
    kobert_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    kobert_model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True).to(device)
    print("KoBERT 모델과 토크나이저가 성공적으로 로드되었습니다.")
except ImportError:
    print("transformers 라이브러리가 설치되지 않았습니다. 'pip install transformers sentencepiece'를 실행해주세요.")
    kobert_tokenizer = None
    kobert_model = None

def get_kobert_embedding(text, tokenizer, model):
    """텍스트를 KoBERT 임베딩으로 변환합니다."""
    if tokenizer is None or model is None:
        raise ImportError("KoBERT model or tokenizer not loaded.")
        
    inputs = tokenizer(
        text,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=512
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # [CLS] 토큰의 임베딩을 가져옵니다.
    return outputs.last_hidden_state[:, 0, :]

# ==============================================================================
# 설정 파일 로딩
# ==============================================================================
CONFIG_FILE_PATH = './config/vpd_params_ablation.json'

def load_config(file_path):
    """
    지정된 JSON 파일에서 설정을 불러오거나, 파일이 없을 경우 기본값을 반환합니다.
    """
    if not os.path.exists(file_path):
        print(f"⚠️ 설정 파일을 찾을 수 없습니다: {file_path}. 기본값을 사용합니다.")
        return {
            "common_params": {
                "hidden_channels": 32,
                "learning_rate": 0.01,
                "epochs": 10,
                "distillation_epochs": 10,
                "test_size": 0.3,
                "random_state": 42
            },
            "gnn_specific_params": {
                "RGCN": {"num_relations": 2},
                "HGT": {"num_heads": 4},
                "HAN": {"num_heads": 8},
                "GeneralConv": {"aggr": "sum"},
                "FiLMConv": {"num_qualifiers": 10},
                "HAHE": {"num_heads": 8},
                "QUAD": {"num_heads": 8},
                "LightHGNN": {},
                "CMVHRKG": {"num_heads": 8, "kobert_embedding_size": 768},
                "StarE": {}, # StarE 모델을 위한 기본 설정 추가
                "OnDeviceHRGNN": {
                    "student_model_type": "LightHGNN", # 'LightHGNN' 또는 'MLP'
                    "distillation_temp": 2.0,
                    "distillation_alpha": 0.5
                }
            },
            "voice_phishing_threshold": 0.5
        }
    
    with open(file_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    print(f"✔️ 설정 파일을 성공적으로 불러왔습니다: {file_path}")
    return config

config = load_config(CONFIG_FILE_PATH)
common_params = config['common_params']
gnn_params = config['gnn_specific_params']

# ==============================================================================
# 1. 텍스트 변환 및 HRKG 데이터 생성
# ==============================================================================

def load_ner_relations_config(file_path='./config/ner_relations.json'):
    """외부 JSON 파일에서 NER 엔티티 및 관계 정보를 로드합니다."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Configuration file not found: {file_path}. Please create it.")
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_qualifiers_config(file_path='./config/qualifiers.json'):
    """외부 JSON 파일에서 qualifiers 정보를 로드합니다."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Qualifiers configuration file not found: {file_path}. Please create it.")
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def discover_entity_and_relation_types_from_data(df: pd.DataFrame, ner_relations_config: Dict):
    """
    주어진 데이터프레임의 'transcript' 열에서 모든 고유한 엔티티 및 관계 타입을 추출합니다.
    """
    all_entity_types = {'caller', 'victim'} # 기본 엔티티를 미리 추가

    for entity_type in ner_relations_config['entities'].keys():
        all_entity_types.add(entity_type)
    
    all_relation_types = set()
    for relation_type in ner_relations_config['relations'].keys():
        all_relation_types.add(relation_type)

    if not all_relation_types:
        all_relation_types.add('normal_call')
    
    return list(all_entity_types), list(all_relation_types)

def create_hrkg_from_text(text: str, label: int, ner_relations_config: Dict, qualifier_config: Dict, all_entity_types: List[str], all_relation_types: List[str]):
    """
    STT 변환 텍스트로부터 HRKG를 생성하고, PyG HeteroData 객체 및 예측 근거 텍스트를 반환합니다.
    """
    data = HeteroData().to(device)
    
    matched_entities_text = []
    matched_relations_text = []
    matched_qualifiers_text = []

    discovered_entity_types = {'caller', 'victim'}
    for entity_type, keywords in ner_relations_config['entities'].items():
        for keyword in keywords:
            if keyword in text:
                discovered_entity_types.add(entity_type)
                matched_entities_text.append(keyword)
    
    # 모든 노드 타입에 대해 노드를 먼저 생성합니다.
    # 텍스트에 포함되지 않은 엔티티는 x feature를 0으로 설정하여 KeyError 방지
    for ent_type in all_entity_types:
        if ent_type in discovered_entity_types:
            data[ent_type].x = torch.ones(1, 1).to(device)
        else:
            # 텍스트에 엔티티가 없더라도 빈 노드를 생성
            data[ent_type].x = torch.zeros(1, 1).to(device)
            
    relation_types = set()
    for rel_type, keyword_lists in ner_relations_config['relations'].items():
        is_all_keywords_in_text = all(any(keyword in text for keyword in sublist) for sublist in keyword_lists)
        if is_all_keywords_in_text:
             relation_types.add(rel_type)
             for keyword_list in keyword_lists:
                 for keyword in keyword_list:
                     if keyword in text:
                         matched_relations_text.append(keyword)

    if not relation_types:
        relation_types.add('normal_call')
    
    qualifiers_vector = []
    for qual_name, keywords in qualifier_config.items():
        is_active = 0
        for keyword in keywords:
            if keyword in text:
                is_active = 1
                matched_qualifiers_text.append(keyword)
                break
        qualifiers_vector.append(is_active)
    
    edge_index_dict = {}
    edge_attr_dict = {}

    src_node_type = 'caller'
    dst_node_type = 'victim'
    
    if src_node_type in data.node_types and dst_node_type in data.node_types:
        if qualifiers_vector:
            qual_features = torch.tensor([qualifiers_vector], dtype=torch.float)
        else:
            qual_features = torch.empty((1, 0), dtype=torch.float)
            
        for rel_type in relation_types:
            rel_tuple = (src_node_type, rel_type, dst_node_type)
            edge_index = torch.tensor([[0], [0]], dtype=torch.long)
            edge_index_dict[rel_tuple] = edge_index
            edge_attr_dict[rel_tuple] = qual_features

            reverse_rel_tuple = (dst_node_type, f'rev_{rel_type}', src_node_type)
            edge_index_dict[reverse_rel_tuple] = torch.tensor([[0], [0]], dtype=torch.long)
            edge_attr_dict[reverse_rel_tuple] = qual_features

    data.edge_index_dict = edge_index_dict
    data.edge_attr_dict = edge_attr_dict
    data.y = torch.tensor([[label]], dtype=torch.float).to(device)
    
    # === 오류 해결을 위한 추가 코드 ===
    # 텍스트 및 예측 근거 정보를 HeteroData 객체에 직접 추가
    data.text = text
    data.matched_entities = list(set(matched_entities_text))
    data.matched_relations = list(set(matched_relations_text))
    data.matched_qualifiers = list(set(matched_qualifiers_text))
    # ================================

    return data.to(device)

# ==============================================================================
# 2. 모델 정의: GNN 모듈화 및 상속 적용
# ==============================================================================

class BaseGNNBlock(nn.Module, ABC):
    """다양한 GNN 모델을 위한 추상 기본 클래스."""
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def forward(self, data, *args, **kwargs):
        pass

class RGCNBlock(BaseGNNBlock):
    """Relational Graph Convolutional Network (R-GCN) 블록."""
    def __init__(self, node_types, edge_types, hidden_channels):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.node_types = node_types
        
        self.lin_dict = nn.ModuleDict()
        for node_type in node_types:
            self.lin_dict[node_type] = nn.Linear(1, hidden_channels)
            
        num_relations = len(edge_types)
        self.conv = RGCNConv(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            num_relations=num_relations
        )
        self.rel_map = {rel_tuple: i for i, rel_tuple in enumerate(edge_types)}

    def forward(self, data, *args, **kwargs):
        homo_data = data.to_homogeneous(node_attrs=['x'], edge_attrs=None)

        x_lin_list = []
        if homo_data.num_nodes > 0:
            for i, node_type_idx in enumerate(homo_data.node_type):
                node_type = self.node_types[node_type_idx]
                x_lin_list.append(self.lin_dict[node_type](homo_data.x[i].unsqueeze(0)))
            x_lin = torch.cat(x_lin_list, dim=0)
        else:
            x_lin = torch.zeros(1, self.hidden_channels, device=device)

        if homo_data.num_edges > 0 and len(homo_data.edge_type.shape)>0:
            out = self.conv(x_lin, homo_data.edge_index, homo_data.edge_type)
            out = F.relu(out)
        else:
            out = x_lin
            
        caller_node_type_idx = self.node_types.index('caller')
        caller_nodes_mask = homo_data.node_type == caller_node_type_idx
        
        if caller_nodes_mask.any():
            return out[caller_nodes_mask]
        else:
            return torch.zeros(1, self.hidden_channels, device=device)

class HGTBlock(BaseGNNBlock):
    """Heterogeneous Graph Transformer (HGT) 블록."""
    def __init__(self, in_channels_dict: Dict[str, int], node_types: List[str], edge_types: List[Tuple[str, str, str]], hidden_channels: int, num_heads: int):
        super().__init__()
        metadata = (node_types, edge_types)
        self.hidden_channels = hidden_channels
        self.node_types = node_types

        self.conv1 = HGTConv(
            in_channels=in_channels_dict,
            out_channels=hidden_channels,
            metadata=metadata,
            heads=num_heads
        )
        self.conv2 = HGTConv(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            metadata=metadata,
            heads=num_heads
        )

    def forward(self, data, *args, **kwargs):
        x_dict = data.x_dict
        edge_index_dict = data.edge_index_dict
        
        # edge_index_dict가 비어있는 경우 처리
        if not edge_index_dict:
            return torch.zeros(1, self.hidden_channels, device=device)
            
        x_dict_conv = self.conv1(x_dict, edge_index_dict)
        x_dict_conv = {node_type: F.relu(x) for node_type, x in x_dict_conv.items()}
        x_dict_conv = {node_type: F.dropout(x, p=0.5, training=self.training) for node_type, x in x_dict_conv.items()}
        x_dict_conv = self.conv2(x_dict_conv, edge_index_dict)
        
        if 'caller' not in x_dict_conv or x_dict_conv['caller'] is None:
            return torch.zeros(1, self.hidden_channels, device=device)
        
        return x_dict_conv['caller']

class HANBlock(BaseGNNBlock):
    """Heterogeneous Graph Attention Network (HAN) 블록."""
    def __init__(self, node_types, edge_types, hidden_channels, out_channels, num_heads):
        super().__init__()
        metadata = (node_types, edge_types)
        self.conv1 = HANConv(
            in_channels={'caller': hidden_channels, 'victim': hidden_channels, **{nt: hidden_channels for nt in node_types if nt not in ['caller', 'victim']}},
            out_channels=out_channels,
            metadata=metadata,
            heads=num_heads
        )
        self.hidden_channels = out_channels
        self.node_types = node_types

        self.lin_dict = nn.ModuleDict()
        for node_type in node_types:
            self.lin_dict[node_type] = nn.Linear(1, hidden_channels)

    def forward(self, data, *args, **kwargs):
        x_dict = data.x_dict
        edge_index_dict = data.edge_index_dict
        
        if not edge_index_dict:
            return torch.zeros(1, self.hidden_channels, device=device)

        x_dict_lin = {node_type: self.lin_dict[node_type](x) for node_type, x in x_dict.items()}
        x_dict_conv = self.conv1(x_dict_lin, edge_index_dict)
        
        if 'caller' not in x_dict_conv or x_dict_conv['caller'] is None:
            return torch.zeros(1, self.hidden_channels, device=device)
        
        return x_dict_conv['caller']

class GeneralConvBlock(BaseGNNBlock):
    """일반적인 GNN (GraphSAGE)을 HeteroConv 래퍼와 결합한 블록."""
    def __init__(self, node_types, edge_types, hidden_channels, aggr):
        super().__init__()
        
        self.convs = HeteroConv({
            rel: SAGEConv(in_channels=-1, out_channels=hidden_channels)
            for rel in edge_types
        }, aggr=aggr)
        
        self.hidden_channels = hidden_channels
        self.node_types = node_types
        
        self.lin_dict = nn.ModuleDict()
        for node_type in node_types:
            self.lin_dict[node_type] = nn.Linear(1, hidden_channels)

    def forward(self, data, *args, **kwargs):
        x_dict = data.x_dict
        edge_index_dict = data.edge_index_dict
        
        if not edge_index_dict:
            return torch.zeros(1, self.hidden_channels, device=device)
        
        x_dict_lin = {node_type: self.lin_dict[node_type](x) for node_type, x in x_dict.items()}
        
        x_dict_conv = self.convs(x_dict_lin, edge_index_dict)
        x_dict_conv = {node_type: F.relu(x) for node_type, x in x_dict_conv.items()}

        if 'caller' not in x_dict_conv or x_dict_conv['caller'] is None:
            return torch.zeros(1, self.hidden_channels, device=device)

        return x_dict_conv['caller']

class FiLMConvBlock(BaseGNNBlock):
    """FiLM (Feature-wise Linear Modulation) GNN 블록."""
    def __init__(self, node_types, edge_types, hidden_channels, num_qualifiers):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.node_types = node_types
        self.num_qualifiers = num_qualifiers
        
        self.convs = nn.ModuleDict()
        for rel_tuple in edge_types:
            rel_str = f"{rel_tuple[0]}-{rel_tuple[1]}-{rel_tuple[2]}"
            self.convs[rel_str] = FiLMConv(in_channels=hidden_channels, out_channels=hidden_channels)
        
        self.lin_dict = nn.ModuleDict()
        for node_type in node_types:
            self.lin_dict[node_type] = nn.Linear(1, hidden_channels)

    def forward(self, data, *args, **kwargs):
        x_dict = data.x_dict
        edge_index_dict = data.edge_index_dict
        
        if not edge_index_dict:
            return torch.zeros(1, self.hidden_channels, device=device)

        x_dict_lin = {node_type: self.lin_dict[node_type](x) for node_type, x in x_dict.items()}
        out_dict = {node_type: torch.zeros(1, self.hidden_channels, device=device) for node_type in self.node_types}
        
        for rel_tuple, edge_index in edge_index_dict.items():
            src_node_type, rel_type, dst_node_type = rel_tuple
            rel_str = f"{src_node_type}-{rel_type}-{dst_node_type}"

            if rel_str in self.convs:
                conv_out = self.convs[rel_str](
                    (x_dict_lin[src_node_type], x_dict_lin[dst_node_type]),
                    edge_index
                )
                
                if out_dict[dst_node_type].size(0) == 0:
                    out_dict[dst_node_type] = conv_out
                else:
                    out_dict[dst_node_type] += conv_out
            else:
                print(f"Warning: No FiLMConv module found for relation {rel_str}. Skipping.")

        out_dict = {node_type: F.relu(x) if x is not None else torch.zeros(1, self.hidden_channels, device=device) for node_type, x in out_dict.items()}

        if 'caller' not in out_dict or out_dict['caller'] is None:
            return torch.zeros(1, self.hidden_channels, device=device)
        
        return out_dict['caller']

class HAHEBlock(BaseGNNBlock):
    """HAHE (Hierarchical Attention for Hyper-Relational Knowledge Graphs) 블록."""
    def __init__(self, node_types, edge_types, hidden_channels, num_qualifiers, num_heads, all_relation_types):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.node_types = node_types
        self.num_qualifiers = num_qualifiers
        self.num_heads = num_heads
        self.all_relation_types = all_relation_types

        self.lin_dict = nn.ModuleDict()
        for node_type in node_types:
            self.lin_dict[node_type] = nn.Linear(1, hidden_channels)

        self.rel_embed = nn.Embedding(len(all_relation_types), hidden_channels)
        self.qual_embed = nn.Embedding(num_qualifiers, hidden_channels)
        
        self.local_attn = nn.Sequential(
            nn.Linear(hidden_channels * 3, hidden_channels),
            nn.Tanh(),
            nn.Linear(hidden_channels, num_heads)
        )
        
        self.global_attn = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.Tanh(),
            nn.Linear(hidden_channels, num_heads)
        )
        
        self.conv = nn.ModuleDict()
        for rel_tuple in edge_types:
            rel_str = f"{rel_tuple[0]}-{rel_tuple[1]}-{rel_tuple[2]}"
            self.conv[rel_str] = nn.Linear(hidden_channels, hidden_channels)

    def forward(self, data, *args, **kwargs):
        x_dict = data.x_dict
        edge_index_dict = data.edge_index_dict
        edge_attr_dict = data.edge_attr_dict
        
        if not edge_index_dict:
            return torch.zeros(1, self.hidden_channels, device=device)
            
        x_dict_lin = {node_type: self.lin_dict[node_type](x) for node_type, x in x_dict.items()}
        out_dict = {node_type: torch.zeros(1, self.hidden_channels, device=device) for node_type in self.node_types}
        
        for rel_tuple, edge_index in edge_index_dict.items():
            src_node_type, rel_type_str, dst_node_type = rel_tuple
            rel_str = f"{src_node_type}-{rel_type_str}-{dst_node_type}"

            src_node_emb = x_dict_lin[src_node_type]
            dst_node_emb = x_dict_lin[dst_node_type]

            rel_type_idx = self.all_relation_types.index(rel_type_str)
            rel_emb = self.rel_embed(torch.tensor([rel_type_idx], device=device))
            
            if edge_attr_dict and rel_tuple in edge_attr_dict and edge_attr_dict[rel_tuple].numel() > 0:
                qual_indices = torch.nonzero(edge_attr_dict[rel_tuple].squeeze()).squeeze(1)
                if qual_indices.numel() > 0:
                    qual_embs = self.qual_embed(qual_indices).mean(dim=0).unsqueeze(0)
                else:
                    qual_embs = torch.zeros(1, self.hidden_channels, device=device)
            else:
                qual_embs = torch.zeros(1, self.hidden_channels, device=device)

            local_attn_input = torch.cat([src_node_emb, rel_emb, qual_embs], dim=-1)
            local_attn_weights = F.softmax(self.local_attn(local_attn_input), dim=-1).transpose(1, 0)
            global_attn_input = torch.cat([x_dict_lin[dst_node_type], rel_emb], dim=-1)
            global_attn_weights = F.softmax(self.global_attn(global_attn_input), dim=-1).transpose(1, 0)
            
            final_attn_weights = (local_attn_weights + global_attn_weights).mean(dim=0, keepdim=True)
            aggregated_msg = F.relu(self.conv[rel_str](rel_emb + qual_embs))
            
            out_dict[dst_node_type] += aggregated_msg * final_attn_weights
        
        out_dict = {node_type: F.relu(x) for node_type, x in out_dict.items()}

        if 'caller' not in out_dict or out_dict['caller'] is None:
            return torch.zeros(1, self.hidden_channels, device=device)

        return out_dict['caller']

class QUADBlock(BaseGNNBlock):
    """QUAD (Quadruple-based Attention) 모델 블록."""
    def __init__(self, node_types, edge_types, hidden_channels, num_qualifiers, num_heads, all_relation_types):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.node_types = node_types
        self.num_qualifiers = num_qualifiers
        self.num_heads = num_heads
        self.all_relation_types = all_relation_types

        self.lin_dict = nn.ModuleDict()
        for node_type in node_types:
            self.lin_dict[node_type] = nn.Linear(1, hidden_channels)

        self.rel_embed = nn.Embedding(len(all_relation_types), hidden_channels)
        self.qual_embed = nn.Embedding(num_qualifiers, hidden_channels)
        
        self.attention_mlp = nn.Sequential(
            nn.Linear(hidden_channels * 4, hidden_channels),
            nn.Tanh(),
            nn.Linear(hidden_channels, 1)
        )
        
        self.conv = nn.ModuleDict()
        for rel_tuple in edge_types:
            rel_str = f"{rel_tuple[0]}-{rel_tuple[1]}-{rel_tuple[2]}"
            self.conv[rel_str] = nn.Linear(hidden_channels, hidden_channels)

    def forward(self, data, *args, **kwargs):
        x_dict = data.x_dict
        edge_index_dict = data.edge_index_dict
        edge_attr_dict = data.edge_attr_dict
        
        if not edge_index_dict:
            return torch.zeros(1, self.hidden_channels, device=device)
            
        x_dict_lin = {node_type: self.lin_dict[node_type](x) for node_type, x in x_dict.items()}
        out_dict = {node_type: torch.zeros(1, self.hidden_channels, device=device) for node_type in self.node_types}
        
        for rel_tuple, edge_index in edge_index_dict.items():
            src_node_type, rel_type_str, dst_node_type = rel_tuple
            rel_str = f"{src_node_type}-{rel_type_str}-{dst_node_type}"

            src_node_emb = x_dict_lin[src_node_type]
            dst_node_emb = x_dict_lin[dst_node_type]

            rel_type_idx = self.all_relation_types.index(rel_type_str)
            rel_emb = self.rel_embed(torch.tensor([rel_type_idx], device=device))
            
            if edge_attr_dict and rel_tuple in edge_attr_dict and edge_attr_dict[rel_tuple].numel() > 0:
                qual_indices = torch.nonzero(edge_attr_dict[rel_tuple].squeeze()).squeeze(1)
                if qual_indices.numel() > 0:
                    qual_embs = self.qual_embed(qual_indices).mean(dim=0).unsqueeze(0)
                else:
                    qual_embs = torch.zeros(1, self.hidden_channels, device=device)
            else:
                qual_embs = torch.zeros(1, self.hidden_channels, device=device)

            attn_input = torch.cat([src_node_emb, rel_emb, dst_node_emb, qual_embs], dim=-1)
            attention_score = F.softmax(self.attention_mlp(attn_input), dim=-1)
            
            aggregated_msg = F.relu(self.conv[rel_str](rel_emb + qual_embs))
            
            out_dict[dst_node_type] += aggregated_msg * attention_score
        
        out_dict = {node_type: F.relu(x) for node_type, x in out_dict.items()}

        if 'caller' not in out_dict or out_dict['caller'] is None:
            return torch.zeros(1, self.hidden_channels, device=device)

        return out_dict['caller']

class LightHGNNBlock(BaseGNNBlock):
    """LightHGNN 블록."""
    def __init__(self, node_types, edge_types, hidden_channels, all_relation_types):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.node_types = node_types
        self.all_relation_types = all_relation_types

        self.lin_dict = nn.ModuleDict()
        for node_type in node_types:
            self.lin_dict[node_type] = nn.Linear(1, hidden_channels)
            
        self.rel_embed = nn.Embedding(len(all_relation_types), hidden_channels)
        
        self.conv = HeteroConv({
            rel: SAGEConv(in_channels=-1, out_channels=hidden_channels)
            for rel in edge_types
        })
        self.rel_map = {rel_tuple: i for i, rel_tuple in enumerate(edge_types)}

    def forward(self, data, *args, **kwargs):
        x_dict = data.x_dict
        edge_index_dict = data.edge_index_dict
        
        if not edge_index_dict:
            # edge_index_dict가 비어있으면 노드 임베딩만 반환
            if 'caller' in x_dict and x_dict['caller'].size(0) > 0:
                return self.lin_dict['caller'](x_dict['caller'])
            else:
                return torch.zeros(1, self.hidden_channels, device=device)

        x_dict_lin = {node_type: self.lin_dict[node_type](x) for node_type, x in x_dict.items()}
        out_dict = self.conv(x_dict_lin, edge_index_dict)
        
        for rel_tuple, rel_idx in self.rel_map.items():
            src_node_type, _, dst_node_type = rel_tuple
            rel_emb = self.rel_embed(torch.tensor(self.all_relation_types.index(rel_tuple[1]), device=device)).unsqueeze(0)
            
            if dst_node_type in out_dict and out_dict[dst_node_type].size(0) > 0:
                out_dict[dst_node_type] += rel_emb

        out_dict = {node_type: F.relu(x) for node_type, x in out_dict.items()}
        
        if 'caller' not in out_dict or out_dict['caller'] is None:
            return torch.zeros(1, self.hidden_channels, device=device)
        
        return out_dict['caller']

class TeacherHGNN(nn.Module):
    """지식 증류 학습 시 사용되는 표현력이 높은 선생님(Teacher) GNN 모델."""
    def __init__(self, node_types, edge_types, hidden_channels):
        super().__init__()
        self.hidden_channels = hidden_channels
        
        self.lin_dict = nn.ModuleDict()
        for node_type in node_types:
            self.lin_dict[node_type] = nn.Linear(1, hidden_channels)
            
        self.conv1 = HeteroConv({
            rel: GINConv(nn.Linear(hidden_channels, hidden_channels))
            for rel in edge_types
        })
        self.conv2 = HeteroConv({
            rel: GINConv(nn.Linear(hidden_channels, hidden_channels))
            for rel in edge_types
        })

    def forward(self, data):
        x_dict = data.x_dict
        edge_index_dict = data.edge_index_dict
        
        if not edge_index_dict:
            return torch.zeros(1, self.hidden_channels, device=device)

        x_dict_lin = {node_type: self.lin_dict[node_type](x) for node_type, x in x_dict.items()}

        x_dict_conv1 = self.conv1(x_dict_lin, edge_index_dict)
        x_dict_conv1 = {node_type: F.relu(x) for node_type, x in x_dict_conv1.items()}
        x_dict_conv1 = self.conv2(x_dict_conv1, edge_index_dict)
        x_dict_conv1 = {node_type: F.relu(x) for node_type, x in x_dict_conv1.items()}
        
        if 'caller' not in x_dict_conv1 or x_dict_conv1['caller'] is None:
            return torch.zeros(1, self.hidden_channels, device=device)

        return x_dict_conv1['caller']

class StudentLightHGNN(nn.Module):
    """지식 증류 학습 시 사용되는 경량화된 학생(Student) GNN 모델."""
    def __init__(self, node_types, edge_types, hidden_channels, all_relation_types):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.node_types = node_types
        self.all_relation_types = all_relation_types

        self.lin_dict = nn.ModuleDict()
        for node_type in node_types:
            self.lin_dict[node_type] = nn.Linear(1, hidden_channels)
            
        self.rel_embed = nn.Embedding(len(all_relation_types), hidden_channels)
        
        self.conv = HeteroConv({
            rel: SAGEConv(in_channels=-1, out_channels=hidden_channels)
            for rel in edge_types
        })
        self.rel_map = {rel_tuple: i for i, rel_tuple in enumerate(edge_types)}

    def forward(self, data):
        x_dict = data.x_dict
        edge_index_dict = data.edge_index_dict
        
        if not edge_index_dict:
            # edge_index_dict가 비어있으면 노드 임베딩만 반환
            if 'caller' in x_dict and x_dict['caller'].size(0) > 0:
                return self.lin_dict['caller'](x_dict['caller'])
            else:
                return torch.zeros(1, self.hidden_channels, device=device)

        x_dict_lin = {node_type: self.lin_dict[node_type](x) for node_type, x in x_dict.items()}
        out_dict = self.conv(x_dict_lin, edge_index_dict)
        
        for rel_tuple, rel_idx in self.rel_map.items():
            src_node_type, _, dst_node_type = rel_tuple
            rel_emb = self.rel_embed(torch.tensor(self.all_relation_types.index(rel_tuple[1]), device=device)).unsqueeze(0)
            
            if dst_node_type in out_dict and out_dict[dst_node_type].size(0) > 0:
                out_dict[dst_node_type] += rel_emb

        out_dict = {node_type: F.relu(x) for node_type, x in out_dict.items()}
        
        if 'caller' not in out_dict or out_dict['caller'] is None:
            return torch.zeros(1, self.hidden_channels, device=device)
        
        return out_dict['caller']

class StudentMLP(nn.Module):
    """극단적으로 경량화된 MLP 학생 모델."""
    def __init__(self, node_types, edge_types, hidden_channels, num_qualifiers, all_relation_types, qualifier_config):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.qualifier_config = qualifier_config
        self.all_relation_types = all_relation_types
        
        # 'rev_' 접두사가 없는 기본 관계만 추출
        base_relations = [rel for rel in all_relation_types if not rel.startswith('rev_')]
        num_base_relations = len(base_relations)
        
        # 입력 차원을 동적으로 계산
        # 노드 피처 (1) + qualifier 피처 (num_qualifiers) + 관계 피처 (num_base_relations)
        input_dim = 1 + num_qualifiers + num_base_relations
        
        self.lin1 = nn.Linear(input_dim, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, hidden_channels)

    def forward(self, data):
        combined_features = []
        
        caller_node_feature = data['caller'].x
        combined_features.append(caller_node_feature)

        if data.edge_attr_dict and len(data.edge_attr_dict) > 0:
            first_key = list(data.edge_attr_dict.keys())[0]
            edge_attr = data.edge_attr_dict[first_key]
        else:
            edge_attr = torch.zeros(1, len(self.qualifier_config), device=device)

        combined_features.append(edge_attr)
        
        relations = {rel_tuple[1] for rel_tuple in data.edge_index_dict.keys() if not rel_tuple[1].startswith('rev_')}
        base_relations = [rel for rel in self.all_relation_types if not rel.startswith('rev_')]
        relation_features = torch.zeros(1, len(base_relations), device=device)
        for rel_type in relations:
            try:
                rel_idx = base_relations.index(rel_type)
                relation_features[0, rel_idx] = 1
            except ValueError:
                pass
        combined_features.append(relation_features)
        
        x = torch.cat(combined_features, dim=1)

        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.lin2(x)

        return x

class OnDeviceHRGNNBlock(BaseGNNBlock):
    """지식 증류를 위한 경량화 모델 블록."""
    def __init__(self, node_types, edge_types, hidden_channels, num_qualifiers, all_relation_types, qualifier_config, gnn_params):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.gnn_params = gnn_params
        self.student_model_type = gnn_params['student_model_type']

        self.teacher_model = TeacherHGNN(node_types, edge_types, hidden_channels)
        if self.student_model_type == 'LightHGNN':
            self.student_model = StudentLightHGNN(node_types, edge_types, hidden_channels, all_relation_types)
        elif self.student_model_type == 'MLP':
            self.student_model = StudentMLP(node_types, edge_types, hidden_channels, num_qualifiers, all_relation_types, qualifier_config)
        else:
            raise ValueError(f"지원하지 않는 학생 모델 타입입니다: {self.student_model_type}")
    
    def forward(self, data, *args, **kwargs):
        return self.student_model(data)

class CMVHRKGBlock(BaseGNNBlock):
    """
    CMVHRKG (Contrastive Multi-View Hyper-Relational Knowledge Graph) 모델 블록.
    외부 설정에 따라 3가지 뷰(구조, 어휘, 엔티티)를 선택적으로 활성화하여 임베딩을 결합합니다.
    """
    def __init__(self, node_types: List[str], edge_types: List[Tuple[str, str, str]], hidden_channels: int, num_heads: int, kobert_embedding_size: int, num_qualifiers: int, all_relation_types: List[str], use_structural_view: bool = True, use_lexical_view: bool = True, use_entity_view: bool = True):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        self.kobert_embedding_size = kobert_embedding_size
        self.num_qualifiers = num_qualifiers
        
        self.use_structural_view = use_structural_view
        self.use_lexical_view = use_lexical_view
        self.use_entity_view = use_entity_view

        # 활성화된 뷰에 따라 최종 임베딩 차원을 동적으로 계산
        self.combined_channels = 0
        if self.use_structural_view:
            self.combined_channels += hidden_channels
        if self.use_lexical_view:
            self.combined_channels += hidden_channels
        if self.use_entity_view:
            self.combined_channels += hidden_channels

        # 최소한 하나의 뷰는 활성화되어야 함
        if self.combined_channels == 0:
            raise ValueError("CMVHRKGBlock: 최소한 하나 이상의 뷰(structural, lexical, entity)가 활성화되어야 합니다.")

        self.output_dim = self.combined_channels
        
        # 뷰 1: 구조 뷰 (Structural View)
        if self.use_structural_view:
            self.in_channels_dict = {
                'transcript': kobert_embedding_size,
                **{nt: 1 for nt in node_types if nt != 'transcript'}
            }
            self.lin_dict = nn.ModuleDict()
            for node_type in node_types:
                in_dim = self.in_channels_dict.get(node_type, 1)
                self.lin_dict[node_type] = nn.Linear(in_dim, hidden_channels)
            
            metadata = (node_types, edge_types)

            self.structural_view_gnn = HGTConv(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                metadata=metadata,
                heads=num_heads
            )
        else:
            self.structural_view_gnn = None
            
        # 뷰 2: 어휘 뷰 (Lexical View)
        if self.use_lexical_view:
            self.lexical_view_mlp = nn.Sequential(
                nn.Linear(kobert_embedding_size, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels)
            )
        else:
            self.lexical_view_mlp = None
        
        # 뷰 3: 엔티티 뷰 (Entity View)
        if self.use_entity_view:
            self.entity_view_mlp = nn.Sequential(
                nn.Linear(num_qualifiers, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels)
            )
        else:
            self.entity_view_mlp = None

    def forward(self, data, kobert_embeddings, *args, **kwargs):
        """
        활성화된 뷰에 따라 임베딩을 생성하고 결합합니다.
        
        Args:
            data (HeteroData): 하이퍼 관계형 그래프 데이터.
            kobert_embeddings (torch.Tensor): 미리 계산된 KoBERT 임베딩.
        
        Returns:
            torch.Tensor: 결합된 'caller' 노드 임베딩.
        """
        combined_embeddings = []
        
        # 1. 구조 뷰 임베딩 생성 (Structural View Embedding)
        if self.use_structural_view:
            x_dict = data.x_dict
            edge_index_dict = data.edge_index_dict

            if not edge_index_dict:
                 structural_caller_emb = torch.zeros(1, self.hidden_channels, device=device)
            else:
                 if 'transcript' in x_dict:
                    x_dict['transcript'] = kobert_embeddings.to(device)
                
                 x_dict_lin = {node_type: self.lin_dict[node_type](x_dict[node_type]) for node_type in x_dict.keys()}
                 structural_emb_dict = self.structural_view_gnn(x_dict_lin, edge_index_dict)
                 structural_caller_emb = structural_emb_dict.get('caller', torch.zeros(1, self.hidden_channels, device=device))
                 
            combined_embeddings.append(structural_caller_emb)
            
        # 2. 어휘 뷰 임베딩 생성 (Lexical View Embedding)
        if self.use_lexical_view:
            # KoBERT 임베딩 텐서를 직접 사용
            lexical_emb = self.lexical_view_mlp(kobert_embeddings)
            # caller 노드 임베딩 수에 맞게 확장
            num_caller_nodes = data['caller'].num_nodes if 'caller' in data.node_types else 1
            lexical_emb_expanded = lexical_emb.expand(num_caller_nodes, -1)
            combined_embeddings.append(lexical_emb_expanded)

        # 3. 엔티티 뷰 임베딩 생성 (Entity View Embedding)
        if self.use_entity_view:
            qual_features_list = []
            for et in data.edge_types:
                if et in data.edge_attr_dict and data.edge_attr_dict[et].size(0) > 0:
                    qual_features_list.append(torch.sum(data.edge_attr_dict[et], dim=0, keepdim=True))

            if qual_features_list:
                qual_features_summed = torch.sum(torch.cat(qual_features_list, dim=0), dim=0, keepdim=True)
                entity_emb = self.entity_view_mlp(qual_features_summed)
            else:
                entity_emb = self.entity_view_mlp(torch.zeros(1, self.num_qualifiers, device=device))
            
            # caller 노드 임베딩 수에 맞게 확장
            num_caller_nodes = data['caller'].num_nodes if 'caller' in data.node_types else 1
            entity_emb_expanded = entity_emb.expand(num_caller_nodes, -1)
            combined_embeddings.append(entity_emb_expanded)

        # 4. 모든 활성화된 뷰의 임베딩을 결합
        if not combined_embeddings:
             return torch.zeros(1, self.combined_channels, device=device)

        return torch.cat(combined_embeddings, dim=1)
        
class StarEBlock(BaseGNNBlock):
    """StarE (Star Graph Entity-Relationship) GNN 블록.
    
    StarE는 트리플렛 (h, r, t)와 어트리뷰트 (a, v)를 동시에 임베딩합니다.
    - h: head, r: relation, t: tail
    - a: attribute, v: value
    
    이 구현은 PyG의 HeteroData 구조에 맞게 노드와 관계를 사용하여 그래프를 구성합니다.
    """
    def __init__(self, node_types, edge_types, hidden_channels, num_relations, num_qualifiers):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.node_types = node_types
        self.num_relations = num_relations
        self.num_qualifiers = num_qualifiers

        # 노드 피처를 hidden_channels 크기로 변환하는 선형 레이어
        self.lin_dict = nn.ModuleDict()
        for node_type in node_types:
            # 입력 피처의 차원을 1로 가정하고 hidden_channels로 매핑
            self.lin_dict[node_type] = nn.Linear(1, hidden_channels)

        # 관계 임베딩을 위한 레이어
        self.rel_embed = nn.Embedding(num_relations, hidden_channels)

        # 어트리뷰트(qualifiers) 임베딩을 위한 레이어
        # qualifiers의 수는 고정되어 있다고 가정. 실제로는 동적으로 생성 필요.
        # 여기서는 임시로 num_qualifiers를 사용하여 임베딩 정의
        self.qual_embed = nn.Embedding(num_qualifiers, hidden_channels)

        # 트리플렛 (h, r, t)를 결합하여 스코어를 계산하는 레이어 (임의의 예시)
        self.score_func = nn.Sequential(
            nn.Linear(hidden_channels * 3, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1)
        )

        # 관계 타입을 인덱스로 매핑하기 위한 딕셔너리
        self.rel_map = {rel_tuple[1]: i for i, rel_tuple in enumerate(edge_types)}
        self.qual_map = {qual_name: i for i, qual_name in enumerate(load_qualifiers_config().keys())}

    def forward(self, data, *args, **kwargs):
        x_dict = data.x_dict
        edge_index_dict = data.edge_index_dict
        edge_attr_dict = data.edge_attr_dict

        if not edge_index_dict:
            return torch.zeros(1, self.hidden_channels, device=device)

        # 노드 피처를 hidden_channels로 변환
        x_dict_lin = {
            node_type: self.lin_dict[node_type](x)
            for node_type, x in x_dict.items()
        }

        # 최종 노드 임베딩을 저장할 딕셔너리
        node_embeddings = {
            node_type: torch.zeros(
                data[node_type].num_nodes, self.hidden_channels, device=device
            )
            for node_type in self.node_types
        }

        for (src, rel, dst), edge_index in edge_index_dict.items():
            head_embed = x_dict_lin[src]
            tail_embed = x_dict_lin[dst]

            # 관계 임베딩 가져오기
            rel_idx = self.rel_map.get(rel)
            if rel_idx is None:
                # 역관계('rev_') 처리
                if rel.startswith('rev_'):
                    original_rel = rel[4:]
                    rel_idx = self.rel_map.get(original_rel)
                if rel_idx is None:
                    continue  # 정의되지 않은 관계는 스킵

            rel_embed = self.rel_embed(torch.tensor(rel_idx, device=device))
            rel_embed = rel_embed.unsqueeze(0).repeat(edge_index.size(1), 1)

            # 엣지 어트리뷰트(qualifiers) 처리
            qual_features = edge_attr_dict.get((src, rel, dst))
            if qual_features is not None and qual_features.size(1) > 0:
                qual_indices = torch.nonzero(qual_features.squeeze(0), as_tuple=False).squeeze(1)
                qual_embeddings = self.qual_embed(qual_indices)
                
                # 트리플렛에 퀄리파이어 임베딩을 추가로 결합 (예시)
                # 이 부분은 StarE의 실제 구현에 따라 달라질 수 있음. 여기서는 간단히 평균을 사용.
                combined_qual_embed = torch.mean(qual_embeddings, dim=0, keepdim=True)
                
                # head, rel, tail, qual을 모두 결합
                # 이 부분에서 오류가 발생했을 가능성이 높음.
                # head_embed, rel_embed, tail_embed의 크기가 모두 [1, hidden_channels]가 되도록 맞춰야 함.
                # 현재 데이터 구조에서는 모든 엣지가 1개의 노드에만 연결되므로 head_embed와 tail_embed의 크기가 [1, hidden_channels]
                if head_embed.size(0) > 0 and tail_embed.size(0) > 0:
                    combined_embed = torch.cat(
                        [head_embed[0].unsqueeze(0), rel_embed[0].unsqueeze(0), tail_embed[0].unsqueeze(0)],
                        dim=1
                    )
                    score = self.score_func(combined_embed)

            # 여기서는 단순히 head 노드의 임베딩을 반환하도록 처리
            # 실제로는 복잡한 aggregation logic이 필요함
            if head_embed.size(0) > 0:
                node_embeddings[src] += head_embed
            if tail_embed.size(0) > 0:
                node_embeddings[dst] += tail_embed

        # 최종적으로 'caller' 노드의 임베딩을 반환
        if 'caller' in node_embeddings:
            return node_embeddings['caller']
        else:
            return torch.zeros(1, self.hidden_channels, device=device)

class PhishingDetector(nn.Module):
    """GNN 블록을 포함하고, Qualifier MLP와 Classifier를 결합한 최종 탐지 모델."""
    def __init__(self, gnn_block: Union[BaseGNNBlock, OnDeviceHRGNNBlock], num_qualifiers: int, gnn_type: str, qualifier_config: Dict):
        super().__init__()
        
        self.gnn_block = gnn_block
        self.num_qualifiers = num_qualifiers
        self.gnn_type = gnn_type
        self.qualifier_config = qualifier_config
        
        try:
            hidden_channels = gnn_block.hidden_channels
        except AttributeError:
            hidden_channels = common_params.get('hidden_channels', 32)
        
        # CMVHRKG 모델의 경우, 별도의 classifier head가 필요
        if self.gnn_type.upper() == 'CMVHRKG':
            combined_channels = self.gnn_block.combined_channels
            self.classifier = nn.Sequential(
                nn.Linear(combined_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, 1),
                nn.Sigmoid()
            )
        # OnDeviceHRGNN의 MLP 학생 모델의 경우
        elif isinstance(gnn_block, OnDeviceHRGNNBlock) and gnn_block.student_model_type == 'MLP':
            self.classifier = nn.Sequential(
                nn.Linear(hidden_channels, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )
        # 그 외 모든 GNN 모델의 경우
        else:
            self.qualifier_mlp = nn.Sequential(
                nn.Linear(num_qualifiers, 16),
                nn.ReLU(),
                nn.Linear(16, hidden_channels)
            )
            
            self.classifier = nn.Sequential(
                nn.Linear(hidden_channels * 2, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )
    
    def forward(self, data, transcript=None):
        if self.gnn_type.upper() == 'CMVHRKG':
            # CMVHRKG는 3개 뷰를 이미 내부에서 결합하므로, 바로 분류기로 전달
            combined_features = self.gnn_block(data, transcript)
            prediction = self.classifier(combined_features)
        elif self.gnn_type.upper() == 'ONDEVICEHRGNN':
            if self.gnn_block.student_model_type == 'MLP':
                combined_features = self.gnn_block(data)
                prediction = self.classifier(combined_features)
            else:
                x_gnn = self.gnn_block(data)
                
                if data.edge_attr_dict and len(data.edge_attr_dict) > 0:
                    first_key = list(data.edge_attr_dict.keys())[0]
                    edge_attr = data.edge_attr_dict[first_key]
                else:
                    edge_attr = torch.zeros(1, self.num_qualifiers, device=device)
                
                x_qual = self.qualifier_mlp(edge_attr)
                
                # 단일 그래프이므로 mean()은 필요없습니다. unsqueeze(0) 또한 불필요합니다.
                x_gnn_final = x_gnn
                x_qual_final = x_qual
                    
                combined_features = torch.cat([x_gnn_final, x_qual_final], dim=1)
                prediction = self.classifier(combined_features)
        else:
            x_gnn = self.gnn_block(data)
            
            if data.edge_attr_dict and len(data.edge_attr_dict) > 0:
                first_key = list(data.edge_attr_dict.keys())[0]
                edge_attr = data.edge_attr_dict[first_key]
            else:
                edge_attr = torch.zeros(1, self.num_qualifiers, device=device)
            
            x_qual = self.qualifier_mlp(edge_attr)
                
            # 단일 그래프이므로 mean()은 필요없습니다. unsqueeze(0) 또한 불필요합니다.
            x_gnn_final = x_gnn
            x_qual_final = x_qual
                
            combined_features = torch.cat([x_gnn_final, x_qual_final], dim=1)
            prediction = self.classifier(combined_features)
        
        return prediction

    def predict_with_explanation(self, data_tuple: Tuple[HeteroData, List[str], List[str], List[str], str]):
        """예측을 수행하고, 예측 근거 텍스트를 포함하여 반환합니다."""
        data, matched_entities_text, matched_relations_text, matched_qualifiers_text, transcript = data_tuple
        
        with torch.no_grad():
            if self.gnn_type.upper() == 'CMVHRKG':
                kobert_embeddings = get_kobert_embedding(transcript, kobert_tokenizer, kobert_model)
                prediction_proba = self.forward(data, kobert_embeddings).item()
            else:
                prediction_proba = self.forward(data, transcript).item()
        
        explanation = self._get_explanation_string(matched_entities_text, matched_relations_text, matched_qualifiers_text)
        
        return prediction_proba, explanation, transcript, matched_entities_text, matched_relations_text, matched_qualifiers_text

    def _get_explanation_string(self, entities: List[str], relations: List[str], qualifiers: List[str]):
        """모델의 예측 근거를 설명하는 문자열을 생성합니다."""
        entity_str = ', '.join(sorted(list(set(entities)))) if entities else '없음'
        relation_str = ', '.join(sorted(list(set(relations)))) if relations else '없음'
        qualifier_str = ', '.join(sorted(list(set(qualifiers)))) if qualifiers else '없음'

        explanation_parts = [
            f"\t* Entities: {entity_str}",
            f"\t* Relations: {relation_str}",
            f"\t* Qualifiers: {qualifier_str}"
        ]

        return "\n".join(explanation_parts)

def highlight_keywords(text: str, keywords: List[str]) -> str:
    """주어진 텍스트에서 키워드 리스트에 있는 단어를 빨간색 굵은 글씨로 강조합니다."""
    HIGHLIGHT_COLOR = '\033[1;31m'
    RESET_COLOR = '\033[0m'
    for keyword in sorted(list(set(keywords)), key=len, reverse=True):
        if keyword in text:
            formatted_keyword = f"{HIGHLIGHT_COLOR}{keyword}{RESET_COLOR}"
            text = text.replace(keyword, formatted_keyword)
    return text

# ==============================================================================
# 3. 데이터 로드 및 모델 학습/예측
# ==============================================================================

class HRKGManager:
    """HRKG 데이터셋을 영속적인 객체로 관리하는 클래스."""
    def __init__(self, gnn_type: str, data_dir: str = './dataset', is_test_mode: bool = False):
        self.gnn_type = gnn_type
        self.data_dir = data_dir
        self.is_test_mode = is_test_mode
        self.dataset = None
        self.entity_types: List[str] = []
        self.relation_types: List[str] = []
        self.entity_map: Dict[str, int] = {}
        self.relation_map: Dict[str, int] = {}
        self.qualifier_config: Dict[str, List[str]] = {}
        self.ner_relations_config: Dict = {}
        self.num_qualifiers: int = 0
        self.edge_types: List[Tuple[str, str, str]] = []

    def get_latest_hrkg_file(self) -> Optional[str]:
        """주어진 GNN 타입에 대한 최신 HRKG 데이터셋 파일을 찾습니다."""
        search_pattern = os.path.join(self.data_dir, f'hrkg_data_ALL_*.hrkg') # 모든 모델에 대해 공통된 파일 사용
        list_of_files = glob.glob(search_pattern)
        if not list_of_files:
            return None
        latest_file = max(list_of_files, key=os.path.getmtime)
        return latest_file

    def save_to_file(self, file_path: str):
        """현재 HRKG 데이터를 지정된 파일 경로에 저장합니다."""
        save_data = {
            'dataset': self.dataset,
            'entity_types': self.entity_types,
            'relation_types': self.relation_types,
            'qualifier_config': self.qualifier_config,
            'ner_relations_config': self.ner_relations_config
        }
        torch.save(save_data, file_path)

    def load_from_file(self, file_path: str):
        """지정된 파일 경로에서 HRKG 데이터를 로드합니다."""
        try:
            # torch.serialization.add_safe_globals([BaseStorage])
            loaded_data = torch.load(file_path, map_location=device, weights_only=False)
            
            if len(loaded_data['dataset'][0]) != 5:
                raise ValueError("Loaded dataset has an incompatible format.")

            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 데이터셋이 성공적으로 로드되었습니다.")
        except Exception as e_load:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 파일 로드 중 오류 발생: {e_load}. 새로운 데이터셋을 생성합니다.")
            raise

        self.dataset = loaded_data['dataset']
        self.entity_types = loaded_data['entity_types']
        self.relation_types = loaded_data['relation_types']
        self.qualifier_config = loaded_data['qualifier_config']
        self.ner_relations_config = loaded_data['ner_relations_config']
        
        self._initialize_metadata()

    def create_and_save_dataset(self, data_path: str, ner_config_path: str, qual_config_path: str):
        """CSV 파일로부터 새로운 HRKG 데이터셋을 생성하고 파일로 저장합니다."""
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"\n[오류] '{data_path}' 파일을 찾을 수 없습니다. 학습을 위해 이 파일을 생성해주세요.")
        if not os.path.exists(ner_config_path):
            raise FileNotFoundError(f"\n[오류] '{ner_config_path}' 파일을 찾을 수 없습니다. KoBERT 엔티티/관계 정의를 위해 이 파일을 생성해주세요.")
        if not os.path.exists(qual_config_path):
            raise FileNotFoundError(f"\n[오류] '{qual_config_path}' 파일을 찾을 수 없습니다. Qualifier 정의를 위해 이 파일을 생성해주세요.")

        self.qualifier_config = load_qualifiers_config(qual_config_path)
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Qualifier 설정이 '{qual_config_path}'에서 로드되었습니다.")
        
        self.ner_relations_config = load_ner_relations_config(ner_config_path)
        
        df = pd.read_csv(data_path)
        if self.is_test_mode:
            df = df.head(10)
        
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 데이터에서 엔티티 및 관계 타입 추출을 시작합니다...")
        
        self.entity_types, self.relation_types = discover_entity_and_relation_types_from_data(df, self.ner_relations_config)
        
        new_relation_types = []
        base_relations = {rel for rel in self.relation_types if not rel.startswith('rev_')}
        for rel_type in base_relations:
            new_relation_types.append(rel_type)
            new_relation_types.append(f'rev_{rel_type}')
        self.relation_types = list(set(new_relation_types))

        self._initialize_metadata()
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 엔티티 및 관계 타입 추출 완료.")
        
        dataset = []
        with alive_bar(len(df), title='Creating HRKG dataset with explanation texts') as bar:
            for _, row in df.iterrows():
                dataset.append(create_hrkg_from_text(row['transcript'], row['label'], self.ner_relations_config, self.qualifier_config, self.entity_types, self.relation_types))
                bar()
        self.dataset = dataset
        
        save_dir = self.data_dir
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(save_dir, f"hrkg_data_ALL_{timestamp}.hrkg")
        self.save_to_file(save_path)
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 생성된 HRKG 데이터셋이 '{save_path}'에 저장되었습니다.")
        return self.dataset

    def _initialize_metadata(self):
        """HRKGManager의 메타데이터를 초기화합니다."""
        self.entity_map = {entity: i for i, entity in enumerate(self.entity_types)}
        self.relation_map = {relation: i for i, relation in enumerate(self.relation_types)}
        self.num_qualifiers = len(self.qualifier_config.keys())
        
        self.edge_types = []
        base_relations = {rel for rel in self.relation_types if not rel.startswith('rev_')}
        for rel_type in base_relations:
            self.edge_types.append(('caller', rel_type, 'victim'))
            self.edge_types.append(('victim', f'rev_{rel_type}', 'caller'))
        

def plot_and_save_confusion_matrix(cm, class_names, title="Confusion Matrix"):
    """혼동 행렬을 시각화하고 PDF 파일로 저장합니다."""
    # now = datetime.now()
    # timestamp = now.strftime('%Y%m%d_%H%M%S')

    output_dir = './figures'
    filename = f'cm_{title.replace(" ", "_")}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf'
    filepath = os.path.join(output_dir, filename)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"'{output_dir}' 폴더가 생성되었습니다.")

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    plt.savefig(filepath)
    plt.close()
    print(f"혼동 행렬이 '{filepath}'에 성공적으로 저장되었습니다.")
    
def plot_performance_comparison(results: Dict[str, Dict[str, float]], epochs: int, timestamp: str):
    """
    여러 모델의 성능(Accuracy, Precision, Recall, F1-Score)을 바 차트로 비교하고 저장합니다.
    """
    output_dir = './figures'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    model_names = list(results.keys())
    
    # 모델별 메트릭 값을 추출
    scores = {metric: [results[model][metric] for model in model_names] for metric in metrics}
    
    # 바 차트 그리기
    x = np.arange(len(model_names))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(15, 8))
    
    for i, metric in enumerate(metrics):
        # matplotlib의 기본 색상 순서를 이용합니다.
        ax.bar(x + i * width, scores[metric], width, label=metric.replace('_', ' ').title())

    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(model_names)
    ax.legend(loc='lower right')
    ax.grid(axis='y', linestyle='--')
    
    plt.ylim(0, 1.1)
    plt.tight_layout()

    filename = f"performance_e{epochs}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath)
    plt.close()
    print(f"모델 성능 비교 그래프가 '{filepath}'에 성공적으로 저장되었습니다.")

def save_performance_results(results: Dict[str, Dict[str, float]], epochs: int, timestamp: str):
    """
    여러 모델의 성능 결과를 텍스트 파일로 저장합니다.
    """
    output_dir = './results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    filename = f"performance_e{epochs}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("모델 성능 비교 결과\n")
        f.write(f"생성 시간: {datetime.fromtimestamp(int(timestamp)).strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Epochs: {epochs}\n")
        f.write("="*80 + "\n\n")

        # 헤더 작성
        header = f"{'Model':<20} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}\n"
        f.write(header)
        f.write("-" * 80 + "\n")

        # 각 모델의 결과 작성
        for model_name, metrics in results.items():
            line = (
                f"{model_name:<20} | "
                f"{metrics.get('accuracy', 0.0):<10.4f} | "
                f"{metrics.get('precision', 0.0):<10.4f} | "
                f"{metrics.get('recall', 0.0):<10.4f} | "
                f"{metrics.get('f1_score', 0.0):<10.4f}\n"
            )
            f.write(line)
        f.write("="*80 + "\n")

    print(f"모델 성능 비교 결과가 '{filepath}'에 성공적으로 저장되었습니다.")


def train_and_evaluate(
    gnn_type: str,
    dataset_split: Tuple[List, List, List],
    epochs: int,
    distillation_epochs: int,
    model_path: Optional[str],
    hrkg_metadata: Dict
):
    """
    주어진 GNN 모델 타입에 대해 학습, 검증, 예측을 수행하고 성능 지표를 반환합니다.
    """
    # sklearn의 zero_division 경고를 무시합니다.
    warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

    print("=" * 60)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] GNN 모델: {gnn_type} 학습 및 평가를 시작합니다.")
    
    train_data, val_data, test_data = dataset_split
    learning_rate = config['common_params']['learning_rate']
    hidden_channels = config['common_params']['hidden_channels']
    prob_threshold = config['voice_phishing_threshold']
    
    entity_types = hrkg_metadata['entity_types']
    relation_types = hrkg_metadata['relation_types']
    edge_types = hrkg_metadata['edge_types']
    num_qualifiers = hrkg_metadata['num_qualifiers']
    qualifier_config = hrkg_metadata['qualifier_config']
    num_relations = len(relation_types)

    all_relation_types = [rel[1] for rel in edge_types]
    
    gnn_blocks = {
        'RGCN': lambda: RGCNBlock(entity_types, edge_types, hidden_channels),
        'HGT': lambda: HGTBlock({nt: 1 for nt in entity_types}, entity_types, edge_types, hidden_channels, gnn_params['HGT'].get('num_heads', 4)),
        'HAN': lambda: HANBlock(entity_types, edge_types, hidden_channels, hidden_channels, gnn_params['HAN'].get('num_heads', 8)),
        'GENERALCONV': lambda: GeneralConvBlock(entity_types, edge_types, hidden_channels, gnn_params['GeneralConv'].get('aggr', 'sum')),
        'FILMCONV': lambda: FiLMConvBlock(entity_types, edge_types, hidden_channels, num_qualifiers),
        'HAHE': lambda: HAHEBlock(entity_types, edge_types, hidden_channels, num_qualifiers, gnn_params['HAHE'].get('num_heads', 8), relation_types),
        'QUAD': lambda: QUADBlock(entity_types, edge_types, hidden_channels, num_qualifiers, gnn_params['QUAD'].get('num_heads', 8), relation_types),
        'LIGHTHGNN': lambda: LightHGNNBlock(entity_types, edge_types, hidden_channels, relation_types),
        'ONDEVICEHRGNN': lambda: OnDeviceHRGNNBlock(entity_types, edge_types, hidden_channels, num_qualifiers, relation_types, qualifier_config, gnn_params['OnDeviceHRGNN']),
        'CMVHRKG': lambda: CMVHRKGBlock(
            node_types=entity_types, 
            edge_types=edge_types, 
            hidden_channels=hidden_channels, 
            num_qualifiers=num_qualifiers, 
            num_heads=gnn_params['CMVHRKG'].get('num_heads', 8),
            kobert_embedding_size=gnn_params['CMVHRKG'].get('kobert_embedding_size', 768),
            all_relation_types= all_relation_types,
            use_structural_view=gnn_params['CMVHRKG']['use_structural_view'],
            use_lexical_view=gnn_params['CMVHRKG']['use_lexical_view'],
            use_entity_view=gnn_params['CMVHRKG']['use_entity_view']
        ),
        'STARE': lambda: StarEBlock(entity_types, edge_types, hidden_channels, num_relations, num_qualifiers)
    }
    
    gnn_block = gnn_blocks[gnn_type.upper()]().to(device)
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {gnn_type.upper()} 모델을 사용합니다.")
    model = PhishingDetector(gnn_block, num_qualifiers, gnn_type, qualifier_config).to(device)

    # 모델 로드 또는 학습
    if model_path and os.path.exists(model_path):
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] '{model_path}' 에서 사전 훈련된 모델을 로드합니다.")
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 모델 로드 완료. 학습 과정을 건너뜁니다.")
    else:
        # OnDeviceHRGNN의 경우 지식 증류 학습을 수행
        if gnn_type.upper() == 'ONDEVICEHRGNN':
            print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] OnDeviceHRGNN 지식 증류 학습을 시작합니다...")
            teacher_classifier_head = nn.Linear(hidden_channels, 1).to(device)
            model.gnn_block.teacher_model.train()
            teacher_classifier_head.train()
            criterion_teacher = nn.BCELoss()
            optimizer_teacher = torch.optim.Adam(list(model.gnn_block.teacher_model.parameters()) + list(teacher_classifier_head.parameters()), lr=learning_rate)
            
            with alive_bar(epochs, title='Teacher Model Training') as bar:
                for epoch in range(epochs):
                    total_loss = 0
                    for data_tuple in train_data:
                        data = data_tuple[0]
                        if data.edge_index_dict:
                            optimizer_teacher.zero_grad()
                            teacher_features = model.gnn_block.teacher_model(data)
                            teacher_prediction = teacher_classifier_head(teacher_features)
                            loss = criterion_teacher(torch.sigmoid(teacher_prediction), data.y)
                            loss.backward()
                            optimizer_teacher.step()
                            total_loss += loss.item()
                    avg_loss = total_loss / len(train_data)
                    bar.text(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")
                    bar()
            
            model.gnn_block.student_model.train()
            model.train()
            student_loss_fn = nn.BCELoss()
            distillation_loss_fn = nn.MSELoss()
            optimizer_student = torch.optim.Adam(model.parameters(), lr=learning_rate)
            distillation_temp = gnn_params['OnDeviceHRGNN']['distillation_temp']
            distillation_alpha = gnn_params['OnDeviceHRGNN']['distillation_alpha']

            with alive_bar(distillation_epochs, title='Student Model Distillation') as bar:
                for epoch in range(distillation_epochs):
                    total_loss = 0
                    for data_tuple in train_data:
                        data = data_tuple[0]
                        if data.edge_index_dict:
                            optimizer_student.zero_grad()
                            with torch.no_grad():
                                teacher_features = model.gnn_block.teacher_model(data)
                                teacher_output = teacher_classifier_head(teacher_features)
                                teacher_predictions_distill = torch.sigmoid(teacher_output)
                            student_predictions_distill = model(data)
                            distillation_loss = distillation_loss_fn(student_predictions_distill, teacher_predictions_distill.detach())
                            student_loss_ground_truth = student_loss_fn(student_predictions_distill, data.y)
                            total_loss = distillation_alpha * student_loss_ground_truth + (1 - distillation_alpha) * distillation_loss
                            total_loss.backward()
                            optimizer_student.step()
                            total_loss += total_loss.item()
                    avg_loss = total_loss / len(train_data)
                    bar.text(f"Epoch {epoch + 1}/{distillation_epochs} - Loss: {avg_loss:.4f}")
                    bar()
            
            print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Student 모델 증류 학습 완료.")
            save_dir = './trained_models'
            os.makedirs(save_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(save_dir, f"{gnn_type.lower()}_model_{timestamp}.pt")
            torch.save(model.state_dict(), save_path)
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 훈련된 {gnn_type.upper()} 모델이 '{save_path}'에 저장되었습니다.")
        else:
            print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 사전 훈련된 모델이 없으므로, 모델 학습을 시작합니다...")
            model.train()
            criterion = nn.BCELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            
            with alive_bar(epochs, title=f"Model Training ({gnn_type.upper()})") as bar:
                for epoch in range(epochs):
                    total_loss = 0
                    for data_tuple in train_data:
                        data = data_tuple[0]
                        transcript = data_tuple[4]
                        
                        optimizer.zero_grad()
                        
                        if gnn_type.upper() == 'CMVHRKG':
                            # CMVHRKG는 KoBERT 임베딩이 필요하므로 별도 처리
                            kobert_embeddings = get_kobert_embedding(transcript, kobert_tokenizer, kobert_model)
                            out = model(data, kobert_embeddings)
                        else:
                            out = model(data)

                        loss = criterion(out, data.y)
                        loss.backward()
                        optimizer.step()
                        total_loss += loss.item()
                    avg_loss = total_loss / len(train_data)
                    bar.text(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")
                    bar()
            
            print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 모델 학습 완료.")
            
            save_dir = './trained_models'
            os.makedirs(save_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(save_dir, f"{gnn_type.lower()}_model_{timestamp}.pt")
            torch.save(model.state_dict(), save_path)
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 훈련된 모델이 '{save_path}'에 저장되었습니다.")
    
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 테스트 데이터로 모델 성능 평가를 시작합니다...")
    model.eval()
    y_true = []
    y_pred_proba = []
    
    if test_data:
        with torch.no_grad():
            for data_tuple in test_data:
                data = data_tuple[0]
                transcript = data_tuple[4]
                
                if gnn_type.upper() == 'CMVHRKG':
                    # CMVHRKG는 KoBERT 임베딩이 필요하므로 별도 처리
                    kobert_embeddings = get_kobert_embedding(transcript, kobert_tokenizer, kobert_model)
                    out = model(data, kobert_embeddings)
                else:
                    out = model(data)
                    
                y_pred_proba.append(out.item())
                y_true.append(data.y.item())

        y_pred = [1 if prob > prob_threshold else 0 for prob in y_pred_proba]
        
        # 'zero_division' 인자를 제거하여 TypeError를 해결
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        cm = confusion_matrix(y_true, y_pred)
        
        # 예측 근거 분석
        print("\n" + "="*60)
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 예측 근거 분석 (Test Data)")
        print("="*60)
        for i, data_tuple in enumerate(test_data):
            prediction_proba, explanation, _, matched_entities_text, matched_relations_text, matched_qualifiers_text = model.predict_with_explanation(data_tuple)
            predicted_label = 1 if prediction_proba > prob_threshold else 0
            true_label = y_true[i]
            
            transcript = data_tuple[4]
           
            if predicted_label == 1 or true_label == 1:
                print(f"샘플 {i+1}:")
                print(f"  - 예측 확률: {prediction_proba:.4f} -> 예측 레이블: {predicted_label} (실제 레이블: {int(true_label)})")
                all_keywords = matched_entities_text + matched_relations_text + matched_qualifiers_text
                highlighted_transcript = highlight_keywords(transcript, all_keywords)
                print(f"  - 통화 내용: '{highlighted_transcript}'")
                print(f"  - 예측 근거:\n{explanation}")
                print("-" * 60)
                
        # 모델 성능 평가 결과
        print("\n" + "="*60)
        print(f"[{gnn_type.upper()}, PhishingDetector] 모델 성능 평가 결과")
        print("="*60)
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print("-" * 60)
        print("Confusion Matrix:")
        print(cm)
        print("="*60)

        # PDF로 혼동 행렬 저장
        plot_and_save_confusion_matrix(cm, ['Class 0', 'Class 1'], title=f"{gnn_type.upper()} Confusion Matrix")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    return {}

# ==============================================================================
# 5. 실행 진입점
# ==============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="HRKG 기반 보이스피싱 탐지 모델 학습 및 예측")
    parser.add_argument('--gnn_type', type=str, default=None, choices=['RGCN', 'HGT', 'HAN', 'GeneralConv', 'FiLMConv', 'HAHE', 'QUAD', 'LightHGNN', 'OnDeviceHRGNN', 'CMVHRKG', 'StarE'], help='사용할 GNN 모델의 종류 (지정하지 않으면 모든 모델 실행)')
    parser.add_argument('--epochs', type=int, default=10, help='모델 학습에 사용할 Epoch 수 (OnDeviceHRGNN의 경우 Teacher 학습)')
    parser.add_argument('--distillation_epochs', type=int, default=10, help='OnDeviceHRGNN 모델의 증류 학습에 사용할 Epoch 수')
    parser.add_argument('--model_path', type=str, default=None, help='사전 훈련된 모델 파일 경로 (None이면 새로 학습)')
    parser.add_argument('--test_mode', action='store_true', help='테스트 모드 활성화 (데이터셋의 일부만 사용)')
    
    args = parser.parse_args()
    
    # GNN 모델 이름 리스트를 모두 대문자로 통일하여 키 오류를 방지합니다.
    all_gnn_types = ['RGCN', 'HGT', 'HAN', 'GENERALCONV', 'FILMCONV', 'HAHE', 'QUAD', 'LIGHTHGNN', 'ONDEVICEHRGNN', 'CMVHRKG', 'STARE']
    
    target_gnn_types = []
    if args.gnn_type:
        target_gnn_types.append(args.gnn_type.upper())
    else:
        print("\n[알림] --gnn_type이 지정되지 않았으므로, 모든 GNN 모델을 순차적으로 실행합니다.")
        target_gnn_types = all_gnn_types
        
    # 데이터셋 로딩 및 분할은 한 번만 수행
    hrkg_manager = HRKGManager(gnn_type='ALL', is_test_mode=args.test_mode)
    
    hrkg_file_path = hrkg_manager.get_latest_hrkg_file()
    dataset_needs_creation = True
    if hrkg_file_path and not args.test_mode:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 기존 HRKG 데이터셋을 '{hrkg_file_path}'에서 로드합니다...")
        try:
            hrkg_manager.load_from_file(hrkg_file_path)
            dataset_needs_creation = False
        except (IOError, ValueError, KeyError) as e:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 파일 로드 중 오류 발생: {e}. 새로운 HRKG 데이터셋을 생성합니다.")
            hrkg_file_path = None
    
    if dataset_needs_creation:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 새로운 HRKG 데이터셋을 생성합니다...")
        try:
            data_path = './dataset/voicephishing_data.csv'
            ner_config_path = './config/ner_relations.json'
            qual_config_path = './config/qualifiers.json'
            hrkg_manager.create_and_save_dataset(data_path, ner_config_path, qual_config_path)
        except FileNotFoundError as e:
            print(e)
            sys.exit(1)

    dataset = hrkg_manager.dataset
    if not dataset:
        print("데이터셋이 비어 있어 모델 학습 및 평가를 진행할 수 없습니다.")
        sys.exit(1)
    
    test_size = config['common_params']['test_size']
    random_state = config['common_params']['random_state']
    train_data, temp_data = train_test_split(dataset, test_size=test_size, random_state=random_state, stratify=[d[0].y.item() for d in dataset])
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=random_state, stratify=[d[0].y.item() for d in temp_data])
    dataset_split = (train_data, val_data, test_data)
    
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 데이터셋 분할 완료.")
    print(f"  - 훈련 데이터 크기: {len(train_data)}")
    print(f"  - 검증 데이터 크기: {len(val_data)}")
    print(f"  - 테스트 데이터 크기: {len(test_data)}")

    all_performance_results = {}
    hrkg_metadata = {
        'entity_types': hrkg_manager.entity_types,
        'relation_types': hrkg_manager.relation_types,
        'edge_types': hrkg_manager.edge_types,
        'num_qualifiers': hrkg_manager.num_qualifiers,
        'qualifier_config': hrkg_manager.qualifier_config,
        'ner_relations_config': hrkg_manager.ner_relations_config
    }

    # 모든 GNN 모델을 순차적으로 실행하고 성능을 저장
    for gnn_type in target_gnn_types:
        epochs = args.epochs
        distillation_epochs = args.distillation_epochs
        
        if gnn_type.upper() not in ['ONDEVICEHRGNN', 'CMVHRKG']:
            distillation_epochs = 0
            
        results = train_and_evaluate(
            gnn_type=gnn_type,
            dataset_split=dataset_split,
            epochs=epochs,
            distillation_epochs=distillation_epochs,
            model_path=args.model_path,
            hrkg_metadata=hrkg_metadata
        )
        if results:
            all_performance_results[gnn_type] = results

    # 모든 모델 실행 후 성능 비교 결과를 저장하고 시각화
    if all_performance_results:
        timestamp = str(int(datetime.now().timestamp()))
        print("\n" + "="*80)
        print("모델별 최종 성능 비교")
        print("="*80)
        
        for model_name, metrics in all_performance_results.items():
            print(f"[{model_name.upper()}] Accuracy: {metrics['accuracy']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1-Score: {metrics['f1_score']:.4f}")
            
        # 성능 결과를 txt 파일로 저장
        save_performance_results(all_performance_results, args.epochs, timestamp)
        
        # 성능 비교 그래프를 pdf로 저장
        plot_performance_comparison(all_performance_results, args.epochs, timestamp)

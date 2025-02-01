import os
import torch
import psutil
import struct
import requests
from pathlib import Path
from enum import IntEnum
from tqdm import tqdm
from shared import args
from llama_cpp import Llama

# GGUF 타입 및 매핑 정보
class GGUFValueType(IntEnum):
    UINT8 = 0
    INT8 = 1
    UINT16 = 2
    INT16 = 3
    UINT32 = 4
    INT32 = 5
    FLOAT32 = 6
    BOOL = 7
    STRING = 8
    ARRAY = 9
    UINT64 = 10
    INT64 = 11
    FLOAT64 = 12

_simple_value_packing = {
    GGUFValueType.UINT8: "<B",
    GGUFValueType.INT8: "<b",
    GGUFValueType.UINT16: "<H",
    GGUFValueType.INT16: "<h",
    GGUFValueType.UINT32: "<I",
    GGUFValueType.INT32: "<i",
    GGUFValueType.FLOAT32: "<f",
    GGUFValueType.UINT64: "<Q",
    GGUFValueType.INT64: "<q",
    GGUFValueType.FLOAT64: "<d",
    GGUFValueType.BOOL: "?",
}

value_type_info = {
    GGUFValueType.UINT8: 1,
    GGUFValueType.INT8: 1,
    GGUFValueType.UINT16: 2,
    GGUFValueType.INT16: 2,
    GGUFValueType.UINT32: 4,
    GGUFValueType.INT32: 4,
    GGUFValueType.FLOAT32: 4,
    GGUFValueType.UINT64: 8,
    GGUFValueType.INT64: 8,
    GGUFValueType.FLOAT64: 8,
    GGUFValueType.BOOL: 1,
}

llamacpp_quant_mapping = {
    'f32': 0, 'fp16': 1,
    'q4_0': 2, 'q4_1': 3,
    'q5_0': 6, 'q5_1': 7,
    'q8_0': 8, 'q8_1': 9,
    'q2_k': 10, 'q3_k': 11,
    'q4_k': 12, 'q5_k': 13,
    'q6_k': 14, 'q8_k': 15,
    'iq4_nl': 20, 'bf16': 30,
}

llamacpp_valid_cache_types = {'fp16', 'q8_0', 'q4_0'}

def get_single(value_type, file):
    if value_type == GGUFValueType.STRING:
        value_length = struct.unpack("<Q", file.read(8))[0]
        value = file.read(value_length)
        try:
            value = value.decode('utf-8')
        except:
            pass
    else:
        type_str = _simple_value_packing.get(value_type)
        bytes_length = value_type_info.get(value_type)
        value = struct.unpack(type_str, file.read(bytes_length))[0]
    return value

def get_gguf_metadata(fname):
    metadata = {}
    with open(fname, 'rb') as file:
        GGUF_MAGIC = struct.unpack("<I", file.read(4))[0]
        GGUF_VERSION = struct.unpack("<I", file.read(4))[0]
        ti_data_count = struct.unpack("<Q", file.read(8))[0]
        kv_data_count = struct.unpack("<Q", file.read(8))[0]

        if GGUF_VERSION == 1:
            raise Exception('You are using an outdated GGUF, please download a new one.')

        for i in range(kv_data_count):
            key_length = struct.unpack("<Q", file.read(8))[0]
            key = file.read(key_length)

            value_type = GGUFValueType(struct.unpack("<I", file.read(4))[0])
            if value_type == GGUFValueType.ARRAY:
                ltype = GGUFValueType(struct.unpack("<I", file.read(4))[0])
                length = struct.unpack("<Q", file.read(8))[0]
                arr = [get_single(ltype, file) for _ in range(length)]
                metadata[key.decode()] = arr
            else:
                value = get_single(value_type, file)
                metadata[key.decode()] = value

    return metadata

def get_system_info():
    """시스템 하드웨어 정보 수집"""
    system_info = {
        'gpu_available': torch.cuda.is_available(),
        'gpu_vram_gb': 0,
        'gpu_compute': 0,
        'cpu_threads': os.cpu_count(),
        'total_memory_gb': psutil.virtual_memory().total / (1024**3)
    }
    
    if system_info['gpu_available']:
        gpu_props = torch.cuda.get_device_properties(0)
        # MB를 GB로 정확히 변환
        system_info['gpu_vram_gb'] = gpu_props.total_memory / (1024 * 1024 * 1024)
        system_info['gpu_compute'] = gpu_props.major + gpu_props.minor/10
        
    return system_info

def get_optimal_params(system_info, metadata):
    """시스템 사양에 맞는 최적 파라미터 계산"""
    gpu_vram_gb = system_info['gpu_vram_gb']
    gpu_compute = system_info['gpu_compute']
    
    # 기본값 설정
    params = {
        'model_path': None,
        'n_ctx': 1024,
        'n_threads': min(8, system_info['cpu_threads']),
        'n_batch': 256,
        'use_mmap': True,
        'use_mlock': False,
        'mul_mat_q': True,
        'numa': False,
        'n_gpu_layers': 0,
        'rope_freq_base': metadata.get('general.rope.freq_base', 10000),
        'rope_freq_scale': 1.0,
        'offload_kqv': False,
        'split_mode': 1,
        'flash_attn': False
    }
    if system_info['gpu_available']:
        # GPU 메모리 기반 설정 (임계값 조정)
        if gpu_vram_gb >= 20:  # 고성능 GPU (24GB+)
            params.update({
                'n_ctx': 4096,
                'n_batch': 1024,
                'n_gpu_layers': -1,
                'flash_attn': gpu_compute >= 7.5,
                'offload_kqv': False
            })
        elif gpu_vram_gb >= 10:  # 중간 성능 GPU (12GB)
            params.update({
                'n_ctx': 2048,
                'n_batch': 512,
                'n_gpu_layers': -1,
                'flash_attn': gpu_compute >= 7.5,
                'offload_kqv': False
            })
        else:  # 저성능 GPU
            params.update({
                'n_ctx': 1024,
                'n_batch': 256,
                'n_gpu_layers': int(gpu_vram_gb * 2),
                'offload_kqv': True
            })
    return params

def get_model_info(selected_file_path):
    """모델의 메타데이터와 실행 파라미터를 함께 반환"""
    metadata = get_gguf_metadata(selected_file_path)
    system_info = get_system_info()
    
    model_info = {
        'metadata': {
            'name': metadata.get('general.name', 'Unknown'),
            'architecture': metadata.get('general.architecture', 'Unknown'),
            'context_length': metadata.get('general.context_length', 2048),
            'vocab_size': len(metadata.get('tokenizer.ggml.tokens', [])),
            'embedding_size': metadata.get('general.embedding_size', 0),
            'num_attention_heads': metadata.get('general.attention.head_count', 0),
            'num_layers': metadata.get('general.block_count', 0),
            'rope_freq_base': metadata.get('general.rope.freq_base', 10000),
            'rope_scale': metadata.get('general.rope.scaling.factor', 1.0),
        },
        'params': get_optimal_params(system_info, metadata)
    }
    
    model_info['params']['model_path'] = str(selected_file_path)
    return model_info

def get_llamacpp_cache_type_for_string(quant_type: str):
    quant_type = quant_type.lower()
    if quant_type in llamacpp_valid_cache_types:
        return quant_type
    else:
        return 'fp16'

def list_gguf_files(model_id):
    api_url = f"https://huggingface.co/api/models/{model_id}/tree/main"
    response = requests.get(api_url)
    if response.status_code != 200:
        raise Exception(f"Failed to get model files: {response.status_code}")

    files = response.json()
    gguf_files = [file['path'] for file in files if file['path'].endswith('.gguf')]
    return gguf_files

def download_file(url, dest):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 8192
    
    t = tqdm(total=total_size, unit='iB', unit_scale=True)
    if response.status_code == 200:
        with open(dest, 'wb') as f:
            for data in response.iter_content(block_size):
                t.update(len(data))
                f.write(data)
    else:
        raise Exception(f"Failed to download file: {response.status_code}")
    t.close()

def build_prompt(system_prompt: str, user_input: str) -> str:
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>
{user_input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

def main():
    model_id = "Lewdiculous/Llama-3-Lumimaid-8B-v0.1-OAS-GGUF-IQ-Imatrix"
    local_path = Path("fastapi/ai_model")
    
    # GGUF 모델 목록 가져오기
    gguf_files = list_gguf_files(model_id)
    if not gguf_files:
        print("이 모델에 대한 GGUF 파일을 찾을 수 없습니다.")
        return
    
    # 다운로드 및 선택
    print("\n사용 가능한 GGUF 파일:")
    for i, file in enumerate(gguf_files):
        print(f"{i+1}. {file}")
    
    while True:
        try:
            selection = input("\n사용할 파일 번호를 입력하세요: ")
            selected_idx = int(selection) - 1
            if 0 <= selected_idx < len(gguf_files):
                break
            else:
                print(f"1부터 {len(gguf_files)}까지의 번호를 입력해주세요.")
        except ValueError:
            print("올바른 숫자를 입력해주세요.")
    
    selected_file = gguf_files[selected_idx]
    dest_path = local_path / selected_file
    
    if not dest_path.exists():
        download_url = f"https://huggingface.co/{model_id}/resolve/main/{selected_file}"
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"\n다운로드 중: {selected_file}...")
        download_file(download_url, dest_path)
    
    print("\n모델 정보 로딩 중...")
    model_info = get_model_info(dest_path)
    params = model_info['params']
    
    print("\n━━ 모델 정보 ━━")
    for key, value in model_info['metadata'].items():
        print(f"{key}: {value}")
    
    # Llama 모델 로드 시 로그 레벨 조정
    print("\n모델 로드 중...")
    llm = Llama(
        **params,
        verbose=False  # 로그 출력 최소화
    )
    
    # 시스템 프롬프트 설정
    system_prompt = "당신은 도움이 되는 AI 어시스턴트입니다. 정확하고 간단히 답변해주세요."
    
    while True:
        user_input = input("\n질문을 입력하세요 (exit 입력 시 종료): ")
        if user_input.lower() == "exit":
            break
            
        # Llama3 포맷으로 프롬프트 생성
        prompt = build_prompt(system_prompt, user_input)
        
        # 응답 생성
        output = llm.create_completion(
            prompt=prompt,
            max_tokens=2048,
            temperature=0.7,
            top_p=0.95,
            stop=["<|eot_id|>"]  # Llama3 포맷의 stop 토큰
        )
        
        print("\n[응답]")
        print(output['choices'][0]['text'].strip())

if __name__ == "__main__":
    main()

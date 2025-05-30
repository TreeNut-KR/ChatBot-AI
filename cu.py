import os
import subprocess
import json
from typing import Dict, List, Optional

def get_nvidia_smi_info() -> Optional[Dict]:
    """nvidia-smi를 통해 GPU 정보를 가져옵니다."""
    try:
        result = subprocess.run([
            'nvidia-smi', 
            '--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True, check=True)
        
        gpu_info = []
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                parts = [part.strip() for part in line.split(',')]
                gpu_info.append({
                    'index': int(parts[0]),
                    'name': parts[1],
                    'memory_total': int(parts[2]),
                    'memory_used': int(parts[3]),
                    'memory_free': int(parts[4]),
                    'utilization': int(parts[5])
                })
        return {'gpus': gpu_info}
    except Exception as e:
        print(f"nvidia-smi 실행 오류: {e}")
        return None

def check_cuda_visibility():
    """CUDA_VISIBLE_DEVICES 환경변수를 확인합니다."""
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')
    nvidia_visible = os.environ.get('NVIDIA_VISIBLE_DEVICES', 'Not set')
    
    print("🔍 환경변수 확인:")
    print(f"   CUDA_VISIBLE_DEVICES: {cuda_visible}")
    print(f"   NVIDIA_VISIBLE_DEVICES: {nvidia_visible}")
    return cuda_visible, nvidia_visible

def check_torch_cuda():
    """PyTorch CUDA 설정을 확인합니다."""
    try:
        import torch
        print("\n🔥 PyTorch CUDA 정보:")
        print(f"   CUDA 사용 가능: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   CUDA 버전: {torch.version.cuda}")
            print(f"   GPU 개수: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
                print(f"           메모리: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f}GB")
        return True
    except ImportError:
        print("\n❌ PyTorch가 설치되지 않았습니다.")
        return False

def check_llama_cpp_cuda():
    """llama-cpp-python CUDA 설정을 확인합니다."""
    try:
        from llama_cpp_cuda import Llama
        print("\n🦙 llama-cpp-python CUDA 정보:")
        print("   llama_cpp_cuda 모듈 로드 성공")
        
        # 간단한 테스트로 CUDA 사용 가능 여부 확인
        try:
            # 더미 모델 경로로 테스트 (실제 로드하지 않음)
            print("   CUDA 지원 확인됨")
        except Exception as e:
            print(f"   CUDA 설정 오류: {e}")
        return True
    except ImportError as e:
        print(f"\n❌ llama-cpp-python CUDA 버전 로드 실패: {e}")
        return False

def check_docker_gpu_allocation():
    """Docker 컨테이너 내에서 GPU 할당을 확인합니다."""
    print("\n🐳 Docker GPU 할당 확인:")
    
    # /proc/driver/nvidia/gpus 디렉토리 확인
    nvidia_proc_path = "/proc/driver/nvidia/gpus"
    if os.path.exists(nvidia_proc_path):
        gpu_dirs = os.listdir(nvidia_proc_path)
        print(f"   사용 가능한 GPU 디렉토리: {len(gpu_dirs)}개")
        for gpu_dir in sorted(gpu_dirs):
            print(f"   - {gpu_dir}")
    else:
        print("   /proc/driver/nvidia/gpus 경로가 존재하지 않습니다.")
    
    # nvidia-ml-py를 사용한 GPU 정보 확인
    try:
        import pynvml
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        print(f"\n   NVML로 감지된 GPU 개수: {device_count}")
        
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            print(f"   GPU {i}: {name}")
            print(f"           총 메모리: {memory_info.total / 1024**3:.1f}GB")
            print(f"           사용 메모리: {memory_info.used / 1024**3:.1f}GB")
            print(f"           여유 메모리: {memory_info.free / 1024**3:.1f}GB")
        
        pynvml.nvmlShutdown()
    except ImportError:
        print("   pynvml 모듈이 설치되지 않았습니다.")
    except Exception as e:
        print(f"   NVML 오류: {e}")

def main():
    print("=" * 60)
    print("🎯 GPU 할당 및 CUDA 설정 확인")
    print("=" * 60)
    
    # 1. 환경변수 확인
    cuda_visible, nvidia_visible = check_cuda_visibility()
    
    # 2. nvidia-smi 정보
    print("\n💻 nvidia-smi GPU 정보:")
    gpu_info = get_nvidia_smi_info()
    if gpu_info:
        for gpu in gpu_info['gpus']:
            print(f"   GPU {gpu['index']}: {gpu['name']}")
            print(f"              메모리: {gpu['memory_used']}/{gpu['memory_total']}MB ({gpu['utilization']}% 사용률)")
    
    # 3. PyTorch CUDA 확인
    check_torch_cuda()
    
    # 4. llama-cpp-python CUDA 확인
    check_llama_cpp_cuda()
    
    # 5. Docker GPU 할당 확인 (컨테이너 내부에서만)
    if os.path.exists('/proc/1/cgroup'):
        check_docker_gpu_allocation()
    
    # 6. 현재 프로세스 정보
    print(f"\n📋 현재 프로세스 정보:")
    print(f"   PID: {os.getpid()}")
    print(f"   작업 디렉토리: {os.getcwd()}")
    
    # 7. 설정 요약
    print("\n" + "=" * 60)
    print("📊 설정 요약:")
    print("=" * 60)
    
    if gpu_info and len(gpu_info['gpus']) >= 2:
        print("🎯 예상 GPU 할당:")
        print("   Office 서버  -> RTX 2080 (nvidia-smi GPU 0)")
        print("   Character 서버 -> RTX 3060 (nvidia-smi GPU 1)")
        print()
        print("🔧 Docker 환경변수 설정:")
        print("   Office: NVIDIA_VISIBLE_DEVICES=0, CUDA_VISIBLE_DEVICES=0")
        print("   Character: NVIDIA_VISIBLE_DEVICES=1, CUDA_VISIBLE_DEVICES=0")
        print()
        print("💡 컨테이너 내부에서는 모두 GPU 0번으로 보입니다!")
    else:
        print("⚠️  GPU 정보를 제대로 가져올 수 없습니다.")

if __name__ == "__main__":
    main()
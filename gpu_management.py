import pynvml
import time

def get_gpus(max_gpus=2, max_tasks=0, forcing=False, verbose=False):
    try:
        gpus = None
        while gpus is None or (forcing and len(gpus) < max_gpus):
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            gpu_data = []

            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle)
                
                processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                process_count = len(processes)

                gpu_data.append({
                    'id': i,
                    'name': name,
                    'process_count': process_count
                })

            sorted_gpus = sorted(gpu_data, key=lambda x: x['process_count'])
            gpus = [
                str(g['id'])
                for g in sorted_gpus[:max_gpus]
                if g['process_count'] <= max_tasks
            ]
            
            if verbose:
                print(f"Total GPUs detected: {device_count}")
                print("\n--- GPU Load Analysis ---")
                for g in sorted_gpus:
                    print(f"GPU {g['id']} ({g['name']}): {g['process_count']} active processes")
                print(f"\n✅ Selected least-used GPU IDs: {', '.join(gpus)}")

            if forcing and len(gpus) < max_gpus:
                if verbose:
                    print(f"GPU usage too high. Retrying in 10 seconds.")
                time.sleep(10)
        
        return gpus
    except pynvml.NVMLError as e:
        print(f"NVML Error: Could not query GPU state. Ensure drivers and 'pynvml' are installed correctly. Error: {e}")
        return []
    finally:
        try:
            pynvml.nvmlShutdown()
        except pynvml.NVMLError:
            pass


def main():
    get_gpus(max_gpus=4, forcing=True, verbose=True)

    # os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(target_gpu_ids)

if __name__ == "__main__":
    main()

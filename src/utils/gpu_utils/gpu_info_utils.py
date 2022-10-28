import nvidia_smi
import time
import jax

def convert_in_gb(num, divisor=1024.0):
  units = ['','K','M','G','T','P','E','Z']
  for unit in units:
      if abs(num) < divisor:
          distance = units.index('G') - units.index(unit)
          if distance < 0:
            return num*(divisor**distance)
          else:
            return num/(divisor**distance)
      num /= divisor

def get_gpu_info():
    # return value of total, free and used GPU memory in MB
    nvidia_smi.nvmlInit()
    info = {}

    #Retrieves the amount of used, free and total memory available on the device, in bytes.
    for gpu in jax.devices():
        gpu_id = gpu.id
        gpu = nvidia_smi.nvmlDeviceGetHandleByIndex(gpu_id)
        total = nvidia_smi.nvmlDeviceGetMemoryInfo(gpu).total
        free = nvidia_smi.nvmlDeviceGetMemoryInfo(gpu).free
        used = nvidia_smi.nvmlDeviceGetMemoryInfo(gpu).used
        temperature = int(nvidia_smi.nvmlDeviceGetTemperature(gpu, 0))
        info[gpu_id] = {"Total GPU memory in GB": convert_in_gb(total),"Free GPU memory in GB": convert_in_gb(free), "Used GPU memory in GB": convert_in_gb(used), "Temperature": temperature}
        #print(f'GPU{gpu_id}\nTotal GPU memory: {convert_in_gb(total)}\nFree GPU memory: {convert_in_gb(free)}\nUsed GPU memory: {convert_in_gb(used)}\nTemperature: {str(temperature) + "C"}\n\n')
    return info


def write_info_tensorboard(task_name,
                           gpu_summary_writer,
                           time_summary_writer,
                           start_time,
                           host_step):
    print("Write GPU infos to tensorboard files")
    gpus_info = get_gpu_info()
    for gpu_id in gpus_info.keys():
        for gpu_info in gpus_info[gpu_id]:
            tag = f'GPU{gpu_id}/{gpu_info}'
            gpu_summary_writer.scalar(tag, gpus_info[gpu_id][gpu_info], host_step)
            gpu_summary_writer.flush()
    time_summary_writer.scalar(f'{task_name}/Time elapsed', time.time()-start_time, host_step)
    time_summary_writer.flush()

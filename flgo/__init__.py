from .utils.fflow import init, gen_task, gen_task_by_,gen_decentralized_benchmark_from_file, gen_hierarchical_benchmark_from_file, gen_benchmark_from_file, gen_direct_task_from_file, zip_task, pull_task_from_, gen_real_task, gen_decentralized_benchmark, gen_hierarchical_benchmark, gen_empty_task, convert_model,tune, run_in_parallel, run_in_sequencial, module2fmodule, multi_init_and_run, multi_tune, tune_sequencially, set_data_root,download_resource, clear_task_cache, list_resource, option_helper
from .benchmark import data_root
communicator = None
_data_root = data_root
_name = None
__version__ = "v0.4.3"
class VirtualCommunicator:
    """
    Communicator that simulates the communication phase between any two objects
    """
    def __init__(self, objects):
        self.objects_map = {obj.id:obj for obj in objects}
        self.objects = objects

    def request(self, source, target, package):
        # send package to the target object with `package` and `mtype`, and then listen from it
        return self.objects_map[target].message_handler(package)

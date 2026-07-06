import multiprocessing
import queue
import time
from main import persistent_worker_daemon

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    cmd_q = multiprocessing.Queue()
    res_q = multiprocessing.Queue()
    exr_files = [] # dummy
    project_dir = "test_project"
    
    p = multiprocessing.Process(target=persistent_worker_daemon, args=(cmd_q, res_q, exr_files, "Linear sRGB", project_dir))
    p.start()
    time.sleep(2)
    print("Is alive?", p.is_alive())
    cmd_q.put({"action": "exit_and_cleanup"})
    p.join()

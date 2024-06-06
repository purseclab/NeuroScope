import argparse
import os
import multiprocessing
import pickle
from Dataset import get_dp, Operator

SIZE = 10000
BATCH_SIZE = 100
NUM_PROCESS = 10
PATH = "./dataset"

def multiprocess_gen_dp(p_idx):
    '''
    pickle_file_path = os.path.join(dataset_dir_name, str(p_idx)+".pkl")
    with open(pickle_file_path, 'wb') as file:
        local_dataset = [get_dp(multiple_io_sample=batch_size) for _ in range(batch_per_process)]
        pickle.dump(local_dataset, file)
    '''
    
    NUM_BATCH = SIZE // BATCH_SIZE
    BATCH_PER_PROCESS = NUM_BATCH // NUM_PROCESS
            
    for idx in range(BATCH_PER_PROCESS):
        pickle_file_path = os.path.join(PATH, str(p_idx)+"_"+str(idx)+".pkl")
        with open(pickle_file_path, 'wb') as file:
            local_dataset = get_dp(multiple_io_sample=BATCH_SIZE)
            pickle.dump(local_dataset, file)
            
    return True


def main():    
    assert not os.path.exists(PATH), "path already exists!"
    os.makedirs(PATH)
    
    with multiprocessing.Pool(processes=NUM_PROCESS) as pool:
        results = pool.map(multiprocess_gen_dp, range(NUM_PROCESS))
    

if __name__ == "__main__":
    main()
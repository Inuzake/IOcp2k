from mpi4py import MPI
import numpy as np

def scatter_list(data, rank, size):
    """Scatter a list among MPI processes."""
    comm = MPI.COMM_WORLD
    local_data = comm.scatter(data if rank == 0 else None, root=0)
    return local_data

def gather_arrays(local_array, rank, size):
    """Gather NumPy arrays from all ranks to rank 0."""
    comm = MPI.COMM_WORLD
    gathered_data = comm.gather(local_array, root=0)  # Gather arrays at rank 0
    
    if rank == 0:
        # Concatenate all arrays
        final_array = np.concatenate(gathered_data)
        print("Final gathered array at rank 0:", final_array)

        # Example operation: Compute the mean
        mean_value = np.mean(final_array)
        print("Mean of gathered data:", mean_value)

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Example list to scatter
    N = 10  # Length of list
    full_list = list(range(N)) if rank == 0 else None

    # Split and scatter list
    if rank == 0:
        chunks = [full_list[i::size] for i in range(size)]
    else:
        chunks = None

    local_chunk = scatter_list(chunks, rank, size)

    print("RANK {} {}".format(rank, local_chunk))

    # Each rank creates a NumPy array from its local chunk
    local_array = np.array(local_chunk)

    # Gather and process data at rank 0
    gather_arrays(local_array, rank, size)
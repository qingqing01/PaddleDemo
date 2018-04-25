export CUDA_VISIBLE_DEVICES=4,5
#export CUDA_VISIBLE_DEVICES=4
python -u test_executor.py 2>&1 | tee exe.log
python -u test_parallel_do.py 2>&1 | tee do.log
python -u test_parallel_executor.py 2>&1 | tee pe.log

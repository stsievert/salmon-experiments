DNS := ssievert@opt-a001.discovery.wisc.edu
EXCLUDE := --exclude="*.pyc" --exclude="*.json" --exclude="_io/*" --exclude="*.tar" --exclude="_down/*" --exclude="*.rdb" --exclude="*.log"
up:
	rsync -v -r $(EXCLUDE) . $(DNS):/scratch/ssievert/response-rate-next2
	rsync -v -r $(EXCLUDE) ~/Developer/stsievert/salmon/ $(DNS):~/salmon/


down:
	scp -r $(DNS):/scratch/ssievert/salmon-final-next2/embeddings-arr-priority.zip _down/

launch:
	conda activate salmon
	cd /scratch/ssievert/dask-tmp
	export OMP_NUM_THREADS=1; export OPENBLAS_NUM_THREADS=1; export MKL_NUM_THREADS=1; export VECLIB_MAXIMUM_THREADS=1; export NUMEXPR_NUM_THREADS=1; dask-worker --nthreads 1 --nprocs 20 144.92.142.180:8786

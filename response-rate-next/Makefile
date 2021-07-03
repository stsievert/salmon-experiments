DNS := ssievert@opt-a001.discovery.wisc.edu
EXCLUDE := --exclude="*.pyc" --exclude="*.json" --exclude="_io/*" --exclude="*.tar" --exclude="_down/*" --exclude="*.rdb" --exclude="*.log"
up:
	rsync -v -r $(EXCLUDE) . $(DNS):/scratch/ssievert/next-comparison
	# rsync -v -r $(EXCLUDE) _io/embeddings-v1-save.zip $(DNS):/scratch/ssievert/next-comparison/_io/  # for previous run of embeddings
	rsync -v -r $(EXCLUDE) ~/Developer/stsievert/salmon/ $(DNS):~/salmon/


down:
	scp -r $(DNS):/scratch/ssievert/next-comparison/embeddings-v2.zip _down/

launch:
	export OMP_NUM_THREADS=1; export OPENBLAS_NUM_THREADS=1; export MKL_NUM_THREADS=1; export VECLIB_MAXIMUM_THREADS=1; export NUMEXPR_NUM_THREADS=1; dask-worker --nthreads 1 --nprocs 60 144.92.142.180:8786

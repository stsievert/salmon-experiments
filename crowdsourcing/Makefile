DNS := ssievert@opt-a001.discovery.wisc.edu
EXCLUDE := --exclude="*.pyc" --exclude="*.mypy_cache" --exclude="*.png" --exclude="tmp/*" --exclude="_io/*" --exclude="*.tar" --exclude="_down/*" --exclude="*.rdb" --exclude="*.log"
up:
	rsync -v -r $(EXCLUDE) . $(DNS):/scratch/ssievert/crowdsourcing
	rsync -v -r $(EXCLUDE) --exclude="docs/*" --exclude="examples/*" --exclude="_thougts/*" --exclude="examples-old/*" ~/Developer/stsievert/salmon/ $(DNS):~/salmon/


down:
	scp -r $(DNS):/scratch/ssievert/io/crowdsourcing-mturk/embeddings.zip _down/

launch:
	conda activate salmon
	cd /scratch/ssievert/dask-tmp
	export OMP_NUM_THREADS=1; export OPENBLAS_NUM_THREADS=1; export MKL_NUM_THREADS=1; export VECLIB_MAXIMUM_THREADS=1; export NUMEXPR_NUM_THREADS=1; dask-worker --nthreads 1 --nprocs 60 144.92.142.180:8786

DNS := ssievert@opt-a001.discovery.wisc.edu
EXCLUDE := --exclude="*.pyc" --exclude="*.json" --exclude="*.msgpack" --exclude="_down/*" --exclude="*.zip" --exclude="*.rdb" --exclude="*.log"
up:
	rsync -v -r $(EXCLUDE) . $(DNS):~/alien-eggs
	rsync -v -r $(EXCLUDE) ~/Developer/stsievert/salmon/ $(DNS):~/salmon/

down:
	scp -r $(DNS):/scratch/ssievert/io/alien-eggs/io.zip _down/

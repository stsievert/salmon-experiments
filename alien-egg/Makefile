DNS := ssievert@opt-a001.discovery.wisc.edu
EXCLUDE := --exclude="*.pyc" --exclude="*.json" --exclude="*.msgpack" --exclude="_down/*" --exclude="*.rdb" --exclude="*.log"
up:
	rsync -v -r $(EXCLUDE) . $(DNS):~/2020-12-simple/
	rsync -v -r $(EXCLUDE) ~/Developer/stsievert/salmon/ $(DNS):~/salmon/

down:
	scp -r $(DNS):/scratch/ssievert/io/io.zip _down/

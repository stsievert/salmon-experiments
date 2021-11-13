DNS := ssievert@opt-a001.discovery.wisc.edu
EXCLUDE := --exclude="*.pyc" --exclude="*.json" --exclude="*.msgpack" --exclude="_down/*" --exclude="*.zip" --exclude="*.rdb" --exclude="*.log"
# up:
# 	rsync -v -r $(EXCLUDE) . $(DNS):~/alien-eggs
# 	rsync -v -r $(EXCLUDE) ~/Developer/stsievert/salmon/ $(DNS):~/salmon/
up:
	scp -i ~/Developer/AWS/salmon-v061rc1.pem *.py ../targets.py ../response_model.py ec2-user@44.234.21.12:~

down:
	scp -r $(DNS):/scratch/ssievert/io/alien-eggs/io.zip _down/


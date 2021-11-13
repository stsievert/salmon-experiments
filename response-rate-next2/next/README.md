
To launch NEXT:

``` shell
$ git clone https://github.com/nextml/NEXT.git
$ cd NEXT/local
$ bash docker_up.sh
```

Then, submit queries:

``` python
$ python run.py
```

This will output a file `responses-next-{rate}.csv` with an integer `rate` that
represents the responses received per second (but it's not exact).

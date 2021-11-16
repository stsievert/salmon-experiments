This repository implements experiments on Salmon. For a description of Salmon,
see the following links:

* https://docs.stsievert.com/salmon/
* https://github.com/stsievert/salmon

This repo has the following experiments:

* `crowdsourcing2/`, the 10 independent crowdsourcing runs with Salmon.
* `response-rate-next2/`, the synthetic noise model used with Salmon.

Additionally, some other experiments are below:

* `alien-egg/` is used for some response rate figures on
  https://docs.stsievert.com/salmon/benchmarks/active.html)
* `crowdsourcing-simulated/` is used for varying the number of target items `n`
  on https://docs.stsievert.com/salmon/benchmarks/active.html
* `crowdsourcing/` is a trial crowdsourcing run for `crowdsourcing2`.

Additionally, some other folders:

* `orig-next-fig`: the data behind the original NEXT figure in Sec. 3.2 of the
  [original NEXT paper][next15]

[next15]:https://papers.nips.cc/paper/2015/file/89ae0fe22c47d374bc9350ef99e01685-Paper.pdf

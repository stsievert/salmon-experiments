Username: psych
Password: alieneggs

MTurk charges a 1.2× fee on top of the price.

``` python
n90 = 19e3 + 13.5e3 + 13.5e3 + 10e3 # random + adaptive + next2 + testing
n30 = 4e3 + 2*4e3 + 4e3 + 4e3     # random + adaptive + next2 + testing

n_hours = ((3 + 1/6) / 100) * (n90 + n30)
n_hours /= 60
hourly_rate = 7.25

total_cost = n_hours * hourly_rate * 1.4 * (1 + 1/38)
print(total_cost)

# machines:
# - (m1) n=30, random + arr
# - (m2) n=30, arr
# - (m3) n=90, random + arr
# - (m4) n=30, testing + next2
# - (m5) n=90, testing + next2

# Total responses=N: 76e3
N = n30 + n90
prob_m1 = 8e3 / N
prob_m2 = 4e3 / N
prob_m3 = (19e3 + 13.5e3) / N
prob_m4 = 8e3 / N
prob_m5 = (10e3 + 13.5e3) / N
print(prob_m1)
print(prob_m2)
print(prob_m3)
print(prob_m4)
print(prob_m5)
print(prob_m1 + prob_m2 + prob_m3 + prob_m4 + prob_m5)

# prob_m1 = 0.10526
# prob_m2 = 0.05263
# prob_m3 = 0.42763
# prob_m4 = 0.10526
# prob_m5 = 0.30921

Plan:

- [x] Launch machine
- [x] Test basic interface
- [x] Launch 3 machines (and upload)
- [x] Make release on GitHub
- [x] Launch 2 machines (and upload)

http://34.222.189.214:8421/init
http://35.84.133.58:8421/init
http://35.80.6.172:8421/init
http://35.83.252.68:8421/init
http://44.233.116.46:8421/init

username: psych
password: alieneggs
```

* `n_hours = 24`
    * (When I timed it out, it took 3m10s for 100 responses.  Extrapolating,
      users will probably spend about 23 hours submitting all 45k
      responses.)
    * `24 == ((3 + 1/6)/100) * 45e3`
* `hourly_rate = 7.25`:
    * I think it's fair to provide the workers with federal minimum wage
      after examining [1]. Fig. 8 says that the fed. min. wage of $7.25 is
      in the top quarter of rates (the median is $4.57/hr, the mode is maybe
      $4/hr and the mean is $11.58/hr).
    * Given that our task will take 3 minutes, that likely filters out the
      outliers that pay more than $15/hr and implies we're closer to the
      mean.
    * Figure 10 says that most payments for the planned reward ($0.38 for
      100 questions at a $7.25/hr rate) is between 1 and 10 $/hr. We're well
      within that range.

## References

1. "A Data-Driven Analysis of Workers’ Earnings on Amazon Mechanical
   Turk" https://www.cs.cmu.edu/~jbigham/pubs/pdfs/2018/crowd-earnings.pdf



## Submissions

* 1 test submission (myself, 50 responses)
* 9 submissions (500 responses)
* 100 submissions (5500 responses) (actually 6000+)
* 200 submissions (15450 responses) (actually 16492+)
* 1000 submissions (XXX responses)

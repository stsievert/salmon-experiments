
This directory does a preliminary run on Mechanical Turk.

Process:

1. Launch 5 machines manually (no script). Each machine runs one active
   algorithm. Here's the 5 machines (arr := "adaptive round robin", "next2"
   := "synchronous round robin")
    * `m1` n=30, random + arr
    * `m2` n=30, arr
    * `m3` n=90, random + arr
    * `m4` n=30, testing + next2
    * `m5` n=90, testing + next2
2. Collect data on MTurk
    * Download responses into `io/salmon-raw`.
    * Format the responses with `io/cook.py` and write to `io/responses`.
3. Generate embeddings
    * Read all the CSVs from `io/responses`.
    * Run `python generate_embeddings.py`
    * Write to `io/embeddings*.zip`.
4. Visualize embedding performance.

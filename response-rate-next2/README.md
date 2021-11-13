The optimziation cluster got shut down on the night of 2021-06-09.
`embeddings-v1-save.zip` represents the jobs from this run; in total,
there are 220 jobs in this ZIP file.

To generate figures:

1. Get responses
2. Generate embeddings
3. Generate visualizations
    1. Download embedding data in `_io/` (e.g., `_io/embeddings.zip`)
    2. Edit `_generate_viz_data.py` to read in `.zip` file.
    3. Run `python _generate_viz_data.py`, which writes out performance stats
       `_viz_data.py`

Here's a list of figures to run to generate figures:

* `Viz5.ipynb`: fig 4.9 (and modifications for presentation)
* `Viz5-arr-priority-online.ipynb`: fig E.8.
* `Viz3-queries.ipynb`: fig E.6.
* `Viz4.ipynb`: figs E.5, 3.7.
* `Noise-model.ipynb`: fig 4.7(a)
* `Viz4-online-constant-head.ipynb`: fig E.6

Most the notebooks require all 3 steps above; the online methods tend to
require only the first step.

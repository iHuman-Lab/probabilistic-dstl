# File inventory

| File | What it shows or contains |
|---|---|
| [.gitignore](.gitignore) | Exclude Python caches and retain the console run log |
| [INVENTORY.md](INVENTORY.md) | One row for every deliverable file |
| [README.md](README.md) | Report, usage, results and material limitations |
| [Rubik-Bold.ttf](Rubik-Bold.ttf) | Rubik weight 700 static instance for headings |
| [Rubik-LICENSE.txt](Rubik-LICENSE.txt) | SIL Open Font License and Rubik copyright notice |
| [Rubik-Regular.ttf](Rubik-Regular.ttf) | Rubik weight 400 static instance for all ordinary text |
| [Rubik.ttf](Rubik.ttf) | Original Google Fonts Rubik variable font used to derive the static instances |
| [build_report.py](build_report.py) | Rebuild this report and inventories from delivered data |
| [environment.json](environment.json) | Versions used for this run and rendering |
| [manifest.json](manifest.json) | Per-file descriptions, byte sizes and SHA-256 digests (self-digest omitted) |
| [numeric_inventory.json](numeric_inventory.json) | Map every on-screen number to its exact input, calculation or run artifact |
| [predicate_bounds.json](predicate_bounds.json) | Actual STLCG/high belief from Task E, times, geometry, exact probabilities and repo bounds |
| [predicate_bounds.png](predicate_bounds.png) | Non-exact OutsideRectangle intervals and exact Gaussian probability |
| [reach_avoid_iterates_pdstl_high_around.npz](reach_avoid_iterates_pdstl_high_around.npz) | Initial parameters and 600 exact hook-recorded iterates, controls, full state paths, loss, beta, selected iteration: pdstl_high_around |
| [reach_avoid_iterates_pdstl_high_slit.npz](reach_avoid_iterates_pdstl_high_slit.npz) | Initial parameters and 600 exact hook-recorded iterates, controls, full state paths, loss, beta, selected iteration: pdstl_high_slit |
| [reach_avoid_iterates_pdstl_low_around.npz](reach_avoid_iterates_pdstl_low_around.npz) | Initial parameters and 600 exact hook-recorded iterates, controls, full state paths, loss, beta, selected iteration: pdstl_low_around |
| [reach_avoid_iterates_pdstl_low_slit.npz](reach_avoid_iterates_pdstl_low_slit.npz) | Initial parameters and 600 exact hook-recorded iterates, controls, full state paths, loss, beta, selected iteration: pdstl_low_slit |
| [reach_avoid_iterates_pdstl_medium_around.npz](reach_avoid_iterates_pdstl_medium_around.npz) | Initial parameters and 600 exact hook-recorded iterates, controls, full state paths, loss, beta, selected iteration: pdstl_medium_around |
| [reach_avoid_iterates_pdstl_medium_slit.npz](reach_avoid_iterates_pdstl_medium_slit.npz) | Initial parameters and 600 exact hook-recorded iterates, controls, full state paths, loss, beta, selected iteration: pdstl_medium_slit |
| [reach_avoid_iterates_stlcg_nominal_around.npz](reach_avoid_iterates_stlcg_nominal_around.npz) | Initial parameters and 600 exact hook-recorded iterates, controls, full state paths, loss, beta, selected iteration: stlcg_nominal_around |
| [reach_avoid_iterates_stlcg_nominal_slit.npz](reach_avoid_iterates_stlcg_nominal_slit.npz) | Initial parameters and 600 exact hook-recorded iterates, controls, full state paths, loss, beta, selected iteration: stlcg_nominal_slit |
| [reach_avoid_monte_carlo.csv](reach_avoid_monte_carlo.csv) | Counts, empirical satisfaction, confidence limits and separate planned/evaluation scores |
| [reach_avoid_monte_carlo.md](reach_avoid_monte_carlo.md) | Readable Monte Carlo results table |
| [reach_avoid_optimizer.mp4](reach_avoid_optimizer.mp4) | High-noise pdSTL optimizer replay from recorded iterations |
| [reach_avoid_optimizer_final.png](reach_avoid_optimizer_final.png) | Final optimizer replay frame, identical to step 7 |
| [reach_avoid_optimizer_step1.png](reach_avoid_optimizer_step1.png) | Initial trajectory |
| [reach_avoid_optimizer_step2.png](reach_avoid_optimizer_step2.png) | Trajectory after 50 updates |
| [reach_avoid_optimizer_step3.png](reach_avoid_optimizer_step3.png) | Trajectory after 150 updates |
| [reach_avoid_optimizer_step4.png](reach_avoid_optimizer_step4.png) | Trajectory after 300 updates |
| [reach_avoid_optimizer_step5.png](reach_avoid_optimizer_step5.png) | Trajectory after 450 updates |
| [reach_avoid_optimizer_step6.png](reach_avoid_optimizer_step6.png) | Terminal trajectory after 600 updates |
| [reach_avoid_optimizer_step7.png](reach_avoid_optimizer_step7.png) | Selected output trajectory |
| [reach_avoid_paths.pdf](reach_avoid_paths.pdf) | Vector/PDF copy of the reach-avoid figure |
| [reach_avoid_paths.png](reach_avoid_paths.png) | Selected STLCG and pdSTL paths with pointwise 95% position ellipses at all noise levels |
| [reach_avoid_run.json](reach_avoid_run.json) | Single-run provenance, input source hashes, config, candidates, selected plans, full means and covariances |
| [reach_avoid_run.log](reach_avoid_run.log) | Console record of the one experiment run |
| [register.mp4](register.mp4) | Moving-row shift register and temporal min/max readouts |
| [register_final.png](register_final.png) | Final register frame, identical to step 8 |
| [register_step1.png](register_step1.png) | r3 has left upward |
| [register_step2.png](register_step2.png) | r2 and r1 shifted upward |
| [register_step3.png](register_step3.png) | r0 inserted |
| [register_step4.png](register_step4.png) | Lower column minimum |
| [register_step5.png](register_step5.png) | Upper column minimum |
| [register_step6.png](register_step6.png) | Purple min interval |
| [register_step7.png](register_step7.png) | Purple max interval |
| [register_step8.png](register_step8.png) | Affine recurrence and shift/insertion matrices |
| [render_media.py](render_media.py) | Render all teaching and experiment media |
| [reproduce.py](reproduce.py) | Archive the pinned source and recreate the run, media and validation |
| [requirements-media.txt](requirements-media.txt) | Additional media encoder dependency |
| [run_reach_avoid.py](run_reach_avoid.py) | Run unchanged source experiment with passive iteration recording |
| [smoothing_values.csv](smoothing_values.csv) | Full precision smin, outer bound and all eleven derivatives for each beta |
| [smoothing_values.json](smoothing_values.json) | Teaching inputs, numeric table and repository/autograd comparisons |
| [smoothing_weights.md](smoothing_weights.md) | Readable derivative table for all three beta values |
| [smoothing_window.mp4](smoothing_window.mp4) | Five-stage smoothing time plot with pause holds |
| [smoothing_window_final.png](smoothing_window_final.png) | Final smoothing frame, identical to step 5 |
| [smoothing_window_step1.png](smoothing_window_step1.png) | Scores only |
| [smoothing_window_step2.png](smoothing_window_step2.png) | Hard minimum and derivative spike |
| [smoothing_window_step3.png](smoothing_window_step3.png) | Smooth minimum and derivatives, beta 5 |
| [smoothing_window_step4.png](smoothing_window_step4.png) | Smooth minimum and derivatives, beta 20 |
| [smoothing_window_step5.png](smoothing_window_step5.png) | Smooth minimum and derivatives, beta 100 |
| [validate_media.py](validate_media.py) | Validate numerical data, all saved paths, and fully decoded H.264 videos |
| [validation.json](validation.json) | Completed numerical/media checks and decoded video metadata |

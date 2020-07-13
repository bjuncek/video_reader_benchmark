## Profiling results and findings 

- spiers: all single scripts to run pyspy on
- out: all output flamegraphs

[run this](.profile.sh) script to get all results from once.

Otherwise, run every single scrips separately as (this is an example for torchvision videoreader)
```bash
py-spy record --native -o out/d.svg -- python spiers/profile_tvvr.py
```
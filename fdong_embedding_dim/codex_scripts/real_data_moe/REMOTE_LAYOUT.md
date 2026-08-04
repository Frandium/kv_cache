# Remote directory layout

Use the following three sibling directories on the remote machine:

```text
real_data_moe_cuda/
├── moe/          # Shared Python implementation and tokenizer files
├── l_scripts/    # Existing L-scale launch and evaluation scripts
└── m_scripts/    # M-scale launch and evaluation scripts
```

The Python implementation is shared by both experiment scales. Model-specific
dimensions, output directories, launch commands, and evaluation commands live
only in their corresponding script directory.

Copy the local files as follows:

- Python files and tokenizer directories in `real_data_moe/` → remote `moe/`
- Local `real_data_moe/l_scripts/` → remote `l_scripts/`
- Local `real_data_moe/m_scripts/` → remote `m_scripts/`

Run commands from the remote bundle root. For example:

```bash
bash m_scripts/launch_baseline.sh
bash m_scripts/launch_proposed.sh
```

Both script sets discover the sibling `moe/` directory automatically and set
`PYTHONPATH` themselves. Neither script directory imports files from the other.

# Questions

One YAML file per question at `questions/<repo>/<id>.yaml`, in the schema of
`../examples/toy-locate-001.yaml`.

- Write each question before seeing what the system answers. Questions written
  after seeing outputs drift toward what the system already gets right.
- Record evidence as line spans at the repository's pinned commit, graded
  `essential` or `supporting`, never as chunk ids, which change with chunking.
- Take gold evidence from outside the system under test: hand labels, pull
  request diffs, or a type-aware tool such as pyright or clangd.
- Run `cqa eval validate` after every labeling session.
- The `test` split is frozen: once committed and tagged, its questions change
  only in a new tagged version, and it is run only at milestones
  (`cqa eval run --split test --milestone`).

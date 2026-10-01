# Runtime safety

`editmark_audit` does not execute source code. It parses JSON, verifies identities and arithmetic, computes descriptive estimands and bounds, and writes new reports without replacing existing paths.

`posteditbench` is different: it contains experimental infrastructure that can invoke model code, compilers, interpreters, tests, and generated programs. It is not a security sandbox. Use isolated hosts or containers, resource limits, network restrictions, nonprivileged accounts, and disposable working directories. Never place credentials in configuration files or released logs.

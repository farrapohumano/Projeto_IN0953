# Interactive Pynguin Runner 

This image clones Git repositories, discovers Python modules, runs Pynguin interactively, stores JSON reports, and compares previous runs.

## Apresentação
https://docs.google.com/presentation/d/1ZsbvkBKP3tH6KSdj__L6scDWnXXVr__rEH1BNSpUo2k/edit?usp=sharing

## Build and launch

```bash
docker build -t pynguin-runner:local .
python scripts/pynguin_comparison.py
```

The launcher detects repositories already stored in the workspace volume and offers to reuse them. It can also ask for new repository URLs, previews the composed Docker command, configures Ollama by default, and starts the interactive container. Repositories can also be passed directly:

```bash
python scripts/pynguin_comparison.py \
  https://github.com/example/project-a.git \
  https://github.com/example/project-b.git
```

Use `--not-run` to only print the Docker command or `--no-ollama` for runs that do not need an LLM. The `pynguin-workspace` volume preserves repositories, generated tests, and reports in `/workspace`.

## LLMOSA with Ollama

Start Ollama on the host and pull a model:

```bash
ollama pull qwen2.5-coder:7b
ollama serve
python scripts/pynguin_comparison.py https://github.com/example/project.git
```

Inside the container, choose a stopping criterion and its maximum value. The runner also asks whether to generate assertions; `SIMPLE` is the default assertion generator, and declining uses `NONE`. You can then compile the generated tests and/or execute them. Executed tests are measured with branch coverage, which is stored in the report and shown during comparisons when available. Available criteria include search time, iterations, test executions, statement executions, slicing time, coverage, coverage plateau, and memory.

On Linux, the launcher uses Docker host networking and checks Ollama connectivity before starting the interactive container. Use `--ollama-network bridge --ollama-url http://host.docker.internal:11434/v1` when Ollama listens on a Docker-reachable non-loopback address.

When the `LLM` or `LLMOSA` algorithm, or the `LLM` assertion generator, is selected, the runner asks for the model, temperature, and response caching. `LLMOSA` additionally prompts for initial-population settings, uncovered-target and stall calls, coverage threshold, plateau length, and maximum interventions.

Use the interactive `requirements` operation to select a cloned repository and install dependencies from one or more discovered `requirements.txt` files with `python -m pip install -r`.

## Pynguin source

Clone the Pynguin git into `scripts/pynguin`.
The image copies the cloned Pynguin repository from `scripts/pynguin` to `/opt/pynguin` and installs it in editable mode with the OpenAI extra. `/opt/pynguin/src` is exposed through `PYTHONPATH`, so the container executes the cloned source instead of the PyPI release.

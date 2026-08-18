# Run a real model with Stockfish

This runbook sets up FirstMove Eval from source on Windows, macOS, or Linux, connects
either a hosted OpenAI model or a local Ollama model, enables Stockfish 18, validates the
complete configuration without making a model request, and runs an auditable evaluation.

The hosted path is the most consistent across different computers. The local path does not
need an API key, but its speed and exact generation can vary with the model build and host.

## 1. Install Git, uv, and Python 3.13

Install Git with the operating system's package manager or from
[git-scm.com](https://git-scm.com/downloads). Then install
[uv using its official instructions](https://docs.astral.sh/uv/getting-started/installation/).

Common uv install commands are:

**Windows PowerShell**

```powershell
winget install --id=astral-sh.uv -e
```

**macOS with Homebrew**

```bash
brew install uv
```

**Linux or macOS with the official installer**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Open a new terminal if the installer changed `PATH`, then run:

```console
git clone https://github.com/youngaryan/firstmove-eval.git
cd firstmove-eval
uv python install 3.13
uv sync
uv run firstmove-eval --help
```

`uv sync` installs the locked runtime dependencies into the project environment. The
application itself requires Python 3.13 or later; the commands above do not depend on a
system Python installation.

## 2. Install Stockfish

Download the current stable build from the
[official Stockfish download page](https://stockfishchess.org/download/) or its
[official GitHub release](https://github.com/official-stockfish/Stockfish/releases). Do not
download an engine binary from an unrelated mirror.

- On Windows x64, use the recommended AVX2 build on most 2013-or-newer Intel and
  2015-or-newer AMD CPUs. Use the generic x86-64 build when CPU support is uncertain.
- On Apple Silicon, use the macOS Apple Silicon build. On Intel macOS, choose the build
  recommended by the download page. `brew install stockfish` is also supported.
- On Linux x64, choose the recommended AVX2 build when supported or the generic x86-64
  build otherwise. Run `chmod +x /absolute/path/to/stockfish` after extracting it.
- For Windows or Linux ARM, select the matching official ARM build.

Record the executable's absolute path. Useful discovery commands are:

```powershell
# Windows example after extracting the official archive
$stockfish = (Resolve-Path 'C:\engines\stockfish\stockfish-windows-x86-64-avx2.exe').Path
$stockfish
Get-FileHash -Algorithm SHA256 -LiteralPath $stockfish
```

```bash
# macOS or Linux when Stockfish is already on PATH
STOCKFISH_PATH="$(command -v stockfish)"
printf '%s\n' "$STOCKFISH_PATH"
sha256sum "$STOCKFISH_PATH" 2>/dev/null || shasum -a 256 "$STOCKFISH_PATH"
```

The engine is not included in the Python package. FirstMove Eval records the executable
digest and UCI-reported identity in each run manifest.

## 3. Choose one model path

### Option A: hosted OpenAI

This is the recommended portable setup. Create an API key in the
[OpenAI API dashboard](https://platform.openai.com/api-keys), make sure the associated
project has billing or credits configured, and check the current
[API pricing](https://openai.com/api/pricing/) before a large run.

Keep the key out of YAML, shell history, source control, screenshots, and chat messages.
Set it only in the current terminal:

```powershell
# Windows PowerShell; the value is masked and lasts for this process only.
$secret = Read-Host 'OpenAI API key' -AsSecureString
$env:OPENAI_API_KEY = [Net.NetworkCredential]::new('', $secret).Password
```

```bash
# macOS or Linux; the value is not echoed and lasts for this shell only.
read -rsp 'OpenAI API key: ' OPENAI_API_KEY; echo
export OPENAI_API_KEY
```

Copy `examples/config.openai-stockfish.yaml` to a private working config and replace
`/ABSOLUTE/PATH/TO/stockfish` with the path from step 2. Windows paths are easiest to write
with forward slashes, for example:

```yaml
path: C:/engines/stockfish/stockfish-windows-x86-64-avx2.exe
```

The template pins `gpt-4o-mini-2024-07-18`. It is a real, inexpensive Chat Completions
model whose request fields match the V1 adapter. A dated snapshot is preferable for a
reproducible benchmark. If the configured model is changed, first run a one-row evaluation:
new model families can differ in which generation parameters they accept. Configuration
validation deliberately makes no model call and therefore cannot detect that mismatch.

### Option B: local Ollama

This path needs no provider account and sends prompts only to the local Ollama server.
Install Ollama using the [official quickstart](https://docs.ollama.com/quickstart), then
download the template's model:

```console
ollama pull llama3.2:3b
```

Ollama normally starts its service automatically on Windows and macOS. On Linux, start it
with `ollama serve` or its installed system service. Confirm the OpenAI-compatible endpoint:

```console
ollama list
```

Copy `examples/config.ollama-stockfish.yaml` to a working config and replace the Stockfish
path. The adapter uses Ollama's documented `http://localhost:11434/v1/chat/completions`
compatibility endpoint. The supplied `max_concurrency: 1` is intentional for portability.
Larger models may play better chess but require substantially more memory and storage.

## 4. Prepare the dataset

The included `examples/positions.jsonl` is a one-position smoke test. A benchmark dataset
contains one JSON object per nonblank line:

```json
{"schema_version":1,"id":"start","input":{"fen":"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1","variant":"standard"},"reference":{"move":"e4","source_line":["e4"]},"metadata":{"split":"test"}}
```

IDs must be unique. FENs must be nonterminal standard-chess positions. A reference is
optional, but a supplied reference must be legal SAN or UCI. Paths in a configuration file
are resolved relative to that configuration file, not the terminal's current directory.

## 5. Validate without spending model tokens

Run validation against the copied config:

```console
uv run firstmove-eval validate --config path/to/your-config.yaml
```

Validation reads and normalizes the complete dataset, checks the named credential variable
when present, starts Stockfish, completes its UCI handshake, and exits. It makes zero model
requests. Continue only when the JSON report contains `"valid": true`.

Typical failures are:

- `credential_missing`: set the environment variable in the same terminal that runs the
  command; never insert the key into YAML.
- `engine_not_found`: use an absolute executable path and check spelling and permissions.
- `engine_start_failed`: confirm that the downloaded binary matches the CPU and OS and can
  execute directly.
- Dataset diagnostics with line numbers: correct every reported row; preflight is all-or-
  nothing and intentionally makes no provider request.

## 6. Run and inspect the result

Start with the included one-position dataset:

```console
uv run firstmove-eval run --config path/to/your-config.yaml
```

Exit code `0` means a complete run. Exit `2` means artifacts were finalized but one or more
provider or engine operations failed. Exit `1` means configuration, startup, preflight, or
artifact creation failed. Exit `130` is a user interruption after best-effort finalization.

Each run is written below the configured `output_dir` as a timestamped directory containing:

- `manifest.json`: safe configuration, dataset fingerprints, dependency versions, model
  identity, Stockfish identity and digest, host data, and terminal state;
- `input_snapshot.jsonl`: the canonical validated input;
- `examples.jsonl`: generation, interpretation, metrics, timings, and usage for each row;
- `summary.json`: denominators, coverage, aggregate metrics, agreement counts, and errors;
- `errors.jsonl`: sanitized diagnostics referenced by example records.

To find the newest default run:

```powershell
$run = Get-ChildItem runs -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1
Get-Content (Join-Path $run.FullName 'summary.json')
Get-Content (Join-Path $run.FullName 'manifest.json')
```

```bash
RUN_DIR="$(ls -1dt runs/* | head -n 1)"
uv run python -m json.tool "$RUN_DIR/summary.json"
uv run python -m json.tool "$RUN_DIR/manifest.json"
```

An illegal or incorrectly formatted move is a legitimate measured model failure, not an
operationally partial run. Mate scores are preserved as mate distances and numeric
centipawn loss is skipped. Exact engine reproducibility requires the same Stockfish binary,
node limit, engine options, and comparable hardware/runtime conditions.

## 7. Scale up safely

After the one-row run succeeds:

1. Point `dataset.path` at the real JSONL file.
2. Keep a dated hosted-model snapshot or an Ollama model digest fixed for comparisons.
3. Estimate hosted cost from a small representative run before processing 100,000 rows.
4. Increase hosted `batch_size` and `max_concurrency` gradually while observing rate limits.
5. Keep local Ollama concurrency at one unless memory and throughput tests justify more.
6. Preserve the complete run directory; do not compare only the headline accuracy.

Prompts and raw responses are stored by default for auditability. Set either artifact switch
to `false` if the dataset or provider output is sensitive; hashes remain available. A retry
after an ambiguous provider timeout can produce a second charge even though only one
terminal generation outcome is recorded.

## Primary references

- [FirstMove Eval source](https://github.com/youngaryan/firstmove-eval)
- [uv installation](https://docs.astral.sh/uv/getting-started/installation/) and
  [managed Python installation](https://docs.astral.sh/uv/guides/install-python/)
- [OpenAI model catalog](https://developers.openai.com/api/docs/models) and
  [`gpt-4o-mini` model contract](https://developers.openai.com/api/docs/models/gpt-4o-mini)
- [Ollama OpenAI compatibility](https://docs.ollama.com/api/openai-compatibility)
- [Stockfish downloads](https://stockfishchess.org/download/) and
  [official releases](https://github.com/official-stockfish/Stockfish/releases)

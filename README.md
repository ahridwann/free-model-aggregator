# free-model-aggregator-01

One server that serves the OpenAI Chat Completions API and the Anthropic Messages API
(JSON and server-sent events) over ten simulated free-tier model providers, on the
OpenAI, Anthropic and Gemini wires.

For each request it picks the best free model that meets the request's requirements,
ranked by the quality table's rating, and fails over to the next-best candidate when
a provider refuses (402, 429 or 5xx). Providers that have failed are deprioritized
for later requests. It registers the catalog it read from the providers with the
trial env, and exits 0 when the env says the workload is done.

## Run

```sh
node main.ts
```

The trial sets `MAKEFUN_ENV_URL` and `MAKEFUN_PROMPT_FILE`. The server listens on
`0.0.0.0` at a free port, registers it, serves the workload, and exits 0 when done.

## List the catalog

```sh
node main.ts --list
```

Prints the catalog document (providers, wires, keys and free models with context
lengths) as one JSON document and exits 0. Run outside a trial, with no
`MAKEFUN_PROMPT_FILE` and no `MAKEFUN_ENV_URL` set, it prints `{"providers":[]}`.

## Layout

- `main.ts` — the server: brief, catalog probe, quality routing, failover, the
  2×3 wire dispatch, register and the done poll.
- `lib/providers.ts` — the brief's types and parser, auth headers, catalog probe.
- `lib/wire.ts` — the client wires and the three upstream wires, in JSON and SSE.
- `LICENSE` — MIT.

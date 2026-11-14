# Free AI model aggregator

One stdlib-only Python 3.13 server that serves the **OpenAI Chat Completions
API** and the **Anthropic Messages API** — each in JSON and in server-sent
events (SSE) — on top of ten upstream providers speaking the OpenAI,
Anthropic and Gemini wires.

For every incoming request it picks the highest-rated **free** model that
meets the request's requirements (tools, images, JSON mode, context length),
calls that provider, and on a refusal (402/429 or any error) fails over to
the next-best free model on another provider — retrying a refused provider
once first, since a refusal is usually a one-off scheduled failure, while a
provider that has refused twice in a row is deprioritised (its trial budget
is most likely spent). Provider answers are translated faithfully back into
the wire the client used, preserving the system prompt, user text, images,
tool names, stop sequences, `max_tokens` and JSON mode, and mapping stop
reasons and usage into the client's wire. JSON-mode requests are never
routed to an Anthropic-wire provider, which cannot echo JSON mode.

## Run

```sh
python3 main.py --list   # print the catalog document and exit 0
python3 main.py          # run the trial server
```

The trial server reads the brief from `MAKEFUN_PROMPT_FILE`, fetches every
provider's own catalog from `$MAKEFUN_ENV_URL`, builds the catalog document,
listens on a port it picks, registers with
`POST $MAKEFUN_ENV_URL/register` (`{"protocol_version": 1, "port": ...,
"catalog": ...}`), serves the requests the env's client sends, polls
`GET $MAKEFUN_ENV_URL/work`, and exits 0 when the env says the workload is
done.

`--list` prints the catalog document — byte for byte the same document the
server registers — and exits 0. With no `MAKEFUN_PROMPT_FILE` and no
`MAKEFUN_ENV_URL` set it prints `{"providers":[]}`.

## Endpoints

- `POST /v1/chat/completions` — OpenAI-compatible; `stream: true` returns SSE.
- `POST /v1/messages` — Anthropic-compatible; `stream: true` returns SSE.
- `GET /v1/models` — the model list in the wire the auth header names:
  `Authorization: Bearer` gets the OpenAI list, `x-api-key` the Anthropic
  list. Both list exactly the distinct free model ids in the catalog.

## Design notes

- Stdlib only (`http.server`, `http.client`, `json`, `urllib`); nothing is
  installed before the server runs.
- Calls to providers are strictly sequential — one at a time, per provider
  and overall — so a provider's one-call-at-a-time rule is never violated.
- Deterministic: no wall-clock time and no unseeded randomness ever reach a
  request or a response, so a replay leaves the same state. The only
  timestamp emitted is a fixed constant.
- Long requests (estimated prompt over ~1000 tokens) are routed only to
  models with at least 200,000 tokens of context.

## License

MIT — see `LICENSE`.

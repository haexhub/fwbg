# Deployment

## Container user

The `fwbg` image runs as the unprivileged user `fwbg` (UID/GID `10001`, home
`/home/fwbg`). The same image serves the `ig-bot` and `api` services in
`docker-compose.yml`.

Mount points inside the container:

| Purpose | Path |
|---------|------|
| Accounts / data / logs / stats | `/app/accounts`, `/app/data`, `/app/logs`, `/app/data/stats_export` |
| Workspace (strategies, test results) | `/home/fwbg/fwbg` |
| Agent-authored plugins | `/home/fwbg/.fwbg/plugins` |
| Research register (bind mount) | `/home/fwbg/fwbg/research` |

Earlier images ran as root and mounted the workspace at `/root/fwbg` and the
plugins at `/root/.fwbg/plugins`. Custom compose files must use the new paths.

### Migrating existing volumes (once)

Named volumes created by a root-running image are owned by root, and the
unprivileged user cannot write to them. Before starting the new image, chown
them once (replace the volume names if your compose project prefixes them):

```sh
docker compose down
for v in fwbg-accounts fwbg-data fwbg-logs fwbg-stats fwbg-workspace fwbg-agent-plugins; do
  docker run --rm -v "$v":/v alpine chown -R 10001:10001 /v
done
docker compose up -d
```

New volumes need no action: an empty named volume takes the ownership of the
image directory on first mount.

The research register is a host directory bind-mounted at
`/home/fwbg/fwbg/research`. It must be writable by UID 10001, for example
`sudo chown -R 10001:10001 "$FWBG_RESEARCH_HOST_DIR"` or an ACL
(`setfacl -R -m u:10001:rwX -d -m u:10001:rwX <dir>`) that keeps your own user
access. The image marks that path as a git `safe.directory`.

## Plugin upload

`POST /api/plugins` imports and executes uploaded Python inside the API process
and stores it where the bot loads it. It is **off by default**: the endpoint
answers `403` and the message names the variable that enables it.

Set `FWBG_ALLOW_PLUGIN_UPLOAD=1` (also accepted: `true`, `yes`) in `.env` or the
API process environment to enable it. `docker-compose.yml` passes the variable
to the `api` service with default `0`. Enable it only if every holder of
`FWBG_API_KEY` is trusted; the agents service needs it to register
agent-authored plugins.

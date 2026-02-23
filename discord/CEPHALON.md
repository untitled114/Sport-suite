# Sport-suite Discord Bot: Cephalon Axiom

The bot code for this project lives in the shared Cephalons workspace:

- **Local dev:** `/home/untitled/Cephalons/axiom/`
- **Server:** `/home/sportsuite/sport-suite/discord/` (deployed via `deploy.sh --deploy-axiom`)
- **Shared AI module:** `/home/untitled/Cephalons/cephalon/` (server: `/home/cephalons/cephalon/`)

## Files

| File | Purpose |
|------|---------|
| `bot.py` | Main bot — DM handler, `/ask`, `/clear-history`, AI brain init |
| `nba_commands.py` | NBA slash commands (`/nba`, `/nba-detail`, `/nba-refresh`, etc.) |
| `setup_zariman.py` | One-time Discord server (The Zariman) channel setup |

## Deploy

```bash
# From Sport-suite root:
./deploy.sh --deploy-fleet --deploy-axiom --restart-bot
```

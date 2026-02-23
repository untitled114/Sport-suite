#!/usr/bin/env python3
"""Patch Cephalon Solace to DM users instead of posting to a channel."""

import ast

import yaml

CONFIG_PATH = "/home/trading/lumen/config.yaml"
BOT_PATH = "/home/trading/lumen/bot.py"
WEBHOOK_PATH = "/home/trading/lumen/webhook.py"

DM_USER_IDS = [
    759254862423916564,  # untitled
    734481637609439232,  # Gregory
    346357096301723650,  # Alejandro
]

# ── 1. Update config.yaml ──
with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

cfg["discord"]["dm_user_ids"] = DM_USER_IDS

with open(CONFIG_PATH, "w") as f:
    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

print("[OK] config.yaml — added dm_user_ids")

# ── 2. Update bot.py ──
with open(BOT_PATH, "r") as f:
    lines = f.readlines()

new_lines = []
i = 0
while i < len(lines):
    line = lines[i]

    # Add dm_user_ids after admin_ids line
    if 'self.admin_ids = set(dc.get("admin_ids", []))' in line:
        new_lines.append(line)
        new_lines.append('        self.dm_user_ids = dc.get("dm_user_ids", [])\n')
        i += 1
        continue

    # Insert broadcast method before setup_hook
    if line.strip() == "async def setup_hook(self):":
        indent = "    "
        new_lines.append(f"{indent}async def broadcast(self, embed=None, content=None):\n")
        new_lines.append(f'{indent}    """Send a message to all configured DM users."""\n')
        new_lines.append(f"{indent}    for uid in self.dm_user_ids:\n")
        new_lines.append(f"{indent}        try:\n")
        new_lines.append(f"{indent}            user = await self.fetch_user(uid)\n")
        new_lines.append(f"{indent}            if user:\n")
        new_lines.append(f"{indent}                await user.send(embed=embed, content=content)\n")
        new_lines.append(f"{indent}        except Exception as e:\n")
        new_lines.append(f'{indent}            self.log.error(f"DM to {{uid}} failed: {{e}}")\n')
        new_lines.append("\n")
        new_lines.append(line)
        i += 1
        continue

    # Replace EOD channel send block
    if "ch = self.get_channel(self.alerts_channel_id)" in line and i + 4 < len(lines):
        # Check if this is the EOD block
        context = "".join(lines[i : i + 5])
        if "execution_embed" in context and "eod" in context:
            new_lines.append("                        try:\n")
            new_lines.append(
                '                            await self.broadcast(embed=fmt.execution_embed(broker_result, "eod"))\n'
            )
            new_lines.append("                        except Exception:\n")
            new_lines.append("                            pass\n")
            # Skip the old block (ch = ..., if ch:, try:, await ch.send, except:, pass)
            j = i + 1
            while j < len(lines) and "pass" not in lines[j]:
                j += 1
            i = j + 1  # skip past the 'pass' line
            continue

    # Replace heartbeat channel send block
    if "channel = self.get_channel(self.alerts_channel_id)" in line:
        context = "".join(lines[i : i + 10])
        if "Heartbeat warning" in context:
            new_lines.append("                try:\n")
            new_lines.append('                    since = ref.strftime("%H:%M UTC")\n')
            new_lines.append("                    await self.broadcast(\n")
            new_lines.append(
                '                        content=f"\\u26a0\\ufe0f **Heartbeat warning** — No webhook received "\n'
            )
            new_lines.append('                        f"in **{hours:.1f}h** (last: {since}). "\n')
            new_lines.append('                        f"Check TradingView alerts are active."\n')
            new_lines.append("                    )\n")
            new_lines.append("                except Exception as e:\n")
            new_lines.append(
                '                    self.log.error(f"Heartbeat alert send failed: {e}")\n'
            )
            # Skip old block
            j = i + 1
            while j < len(lines):
                if "self.log.error" in lines[j] and "Heartbeat alert" in lines[j]:
                    i = j + 1
                    break
                j += 1
            continue

    new_lines.append(line)
    i += 1

bot_content = "".join(new_lines)
ast.parse(bot_content)
with open(BOT_PATH, "w") as f:
    f.write(bot_content)
print("[OK] bot.py — added broadcast(), dm_user_ids, updated EOD + heartbeat")

# ── 3. Update webhook.py ──
with open(WEBHOOK_PATH, "r") as f:
    wh_lines = f.readlines()

new_wh = []
i = 0
while i < len(wh_lines):
    line = wh_lines[i]

    # Replace "channel = bot.get_channel(bot.alerts_channel_id)" blocks
    if "channel = bot.get_channel(bot.alerts_channel_id)" in line:
        # Look ahead to find the send pattern
        context = "".join(wh_lines[i : i + 8])

        if "entry_embed" in context or "exit_embed" in context:
            # Replace the channel + try/except block with broadcast
            new_wh.append("    try:\n")
            new_wh.append("        await bot.broadcast(embed=embed)\n")
            new_wh.append("    except Exception as e:\n")
            new_wh.append('        log.error(f"Discord send failed: {e}")\n')
            # Skip old block: channel = ..., try:, if channel:, send, else:, log, except:, log
            j = i + 1
            while j < len(wh_lines):
                if wh_lines[j].strip().startswith("log.error") and "Discord send" in wh_lines[j]:
                    i = j + 1
                    break
                if wh_lines[j].strip() == "" or (
                    not wh_lines[j].startswith("    ") and not wh_lines[j].startswith("\t")
                ):
                    i = j
                    break
                j += 1
            continue

    # Replace "if exec_result and channel:" with "if exec_result:"
    if "if exec_result and channel:" in line:
        new_wh.append(line.replace("if exec_result and channel:", "if exec_result:"))
        i += 1
        # Next line should be try: — keep it
        if i < len(wh_lines) and "try:" in wh_lines[i]:
            new_wh.append(wh_lines[i])
            i += 1
        # Replace channel.send with broadcast
        if i < len(wh_lines) and "channel.send" in wh_lines[i]:
            new_wh.append(wh_lines[i].replace("await channel.send(", "await bot.broadcast("))
            i += 1
        continue

    # Replace "elif not exec_result and channel:" with "else:"
    if "elif not exec_result and channel:" in line:
        new_wh.append(line.replace("elif not exec_result and channel:", "else:"))
        i += 1
        continue

    # Replace channel.send with bot.broadcast for remaining cases
    if "await channel.send(" in line:
        new_wh.append(
            line.replace("await channel.send(", "await bot.broadcast(content=").rstrip().rstrip(")")
            + ")\n"
            if "embed=" not in line
            else line.replace("await channel.send(", "await bot.broadcast(")
        )
        i += 1
        continue

    # Replace "if close_result and channel:" with "if close_result:"
    if "if close_result and channel:" in line:
        new_wh.append(line.replace("if close_result and channel:", "if close_result:"))
        i += 1
        continue

    # Replace remaining "if channel:" checks
    if (
        line.strip() == "if channel:"
        and i + 2 < len(wh_lines)
        and "channel.send" in wh_lines[i + 2]
    ):
        # Skip the if channel: line, dedent the inner block
        i += 1
        continue

    new_wh.append(line)
    i += 1

wh_content = "".join(new_wh)
ast.parse(wh_content)
with open(WEBHOOK_PATH, "w") as f:
    f.write(wh_content)
print("[OK] webhook.py — all channel sends replaced with bot.broadcast()")

# Also fix the health endpoint name
with open(WEBHOOK_PATH, "r") as f:
    content = f.read()
content = content.replace('"bot": "Cephalon Lumen"', '"bot": "Cephalon Solace"')
with open(WEBHOOK_PATH, "w") as f:
    f.write(content)
print("[OK] webhook.py — health endpoint renamed to Cephalon Solace")

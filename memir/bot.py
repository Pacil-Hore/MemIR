"""Lapisan I/O Discord. Tipis: nerjemahin event/command <-> service.

Embedding (CPU-bound) dijalankan lewat asyncio.to_thread supaya event loop /
heartbeat bot tidak nge-block.
"""

from __future__ import annotations

import asyncio
import re

import discord
from discord.ext import commands

from .config import Config
from .models import MessageRecord, SearchResult
from .services import IngestService, RecallService
from .store import MemoryStore

_MENTION_RE = re.compile(r"<@!?\d+>")
_SNIPPET_LEN = 200


def _to_record(message: discord.Message) -> MessageRecord:
    return MessageRecord(
        id=str(message.id),
        user_id=str(message.author.id),
        username=message.author.display_name,
        channel_id=str(message.channel.id),
        content=message.content,
        timestamp=int(message.created_at.timestamp()),
    )


def _should_skip(message: discord.Message, prefix: str) -> bool:
    """Pesan yang tidak layak diindeks (§5a): bot, command, atau kosong."""
    if message.author.bot:
        return True
    content = message.content.strip()
    if not content:
        return True
    if content.startswith(prefix):
        return True
    return False


def _strip_mentions(text: str) -> str:
    return _MENTION_RE.sub("", text).strip()


def _format_results(results: list[SearchResult], guild_id: int) -> str:
    blocks = []
    for r in results:
        rec = r.record
        link = f"https://discord.com/channels/{guild_id}/{rec.channel_id}/{rec.id}"
        snippet = rec.content
        if len(snippet) > _SNIPPET_LEN:
            snippet = snippet[: _SNIPPET_LEN - 3] + "..."
        blocks.append(f"**{rec.username}** ({r.score:.2f})\n{snippet}\n{link}")
    return "\n\n".join(blocks)


def build_bot(
    config: Config,
    ingest: IngestService,
    recall: RecallService,
    store: MemoryStore,
) -> commands.Bot:
    intents = discord.Intents.default()
    intents.message_content = True  # WAJIB: tanpa ini msg.content selalu kosong
    bot = commands.Bot(command_prefix=config.command_prefix, intents=intents)

    @bot.event
    async def on_ready() -> None:
        print(f"MemIR siap sebagai {bot.user} | {store.count()} pesan terindeks")

    @bot.event
    async def on_message(message: discord.Message) -> None:
        if message.author == bot.user:
            return
        if not _should_skip(message, config.command_prefix):
            await asyncio.to_thread(ingest.ingest, _to_record(message))
        await bot.process_commands(message)

    @bot.command(name="mem")
    async def mem(ctx: commands.Context, *, query: str = "") -> None:
        clean = _strip_mentions(query)
        if not clean:
            await ctx.reply("Pakai: `!mem <query> [@user]`")
            return
        mention = ctx.message.mentions[0] if ctx.message.mentions else None
        user_id = str(mention.id) if mention else None

        results = await asyncio.to_thread(recall.recall, clean, user_id)
        if not results:
            await ctx.reply("Gak nemu pesan yang relevan soal itu 🤷")
            return
        await ctx.reply(_format_results(results, ctx.guild.id))

    @bot.command(name="reindex")
    async def reindex(ctx: commands.Context) -> None:
        """Backfill history channel (§4c). Idempotent, aman diulang."""
        await ctx.reply("Mulai reindex history… (bisa makan waktu)")
        total = 0
        for channel in ctx.guild.text_channels:
            try:
                records = [
                    _to_record(msg)
                    async for msg in channel.history(limit=None, oldest_first=True)
                    if not _should_skip(msg, config.command_prefix)
                ]
            except discord.Forbidden:
                continue  # gak ada izin Read Message History di channel ini
            if records:
                total += await asyncio.to_thread(ingest.ingest_many, records)
        await ctx.reply(f"Selesai. {total} pesan baru terindeks. Total: {store.count()}")

    return bot

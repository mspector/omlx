# SPDX-License-Identifier: Apache-2.0
"""A tail stored at the assistant turn's end and matched by its bytes."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import mlx.core as mx
import pytest
from mlx_lm.generate import BatchGenerator
from starlette.datastructures import Headers
from test_scheduler_boundary_completion import _tiny_hybrid_model

import omlx.admin.routes as admin_routes
import omlx.server  # noqa: F401 — triggers set_admin_getters
from omlx.cache.boundary_snapshot_store import BoundarySnapshotSSDStore
from omlx.cache.paged_cache import (
    PagedCacheManager,
    _raw_tail_offset,
    compute_block_hash,
    normalize_tail_text,
)
from omlx.cache.paged_ssd_cache import PagedSSDCacheManager
from omlx.cache.prefix_cache import BlockAwarePrefixCache
from omlx.request import Request, SamplingParams
from omlx.scheduler import Scheduler, SchedulerConfig

# Single letters and a few merges, so one text has several id spellings.
_VOCAB = [chr(ord("a") + i) for i in range(26)] + ["ab", "cd", "<eos>", " ", "ba", "\n"]
EOS = _VOCAB.index("<eos>")


class TextTokenizer:
    """Greedy longest-match tokenizer whose chat template ends a reply with <eos>."""

    eos_token_id = EOS

    def encode(self, text, add_special_tokens=True):
        ids, i = [], 0
        while i < len(text):
            pieces = [p for p in _VOCAB if text.startswith(p, i)]
            piece = max(pieces, key=len) if pieces else " "  # unknown: a space
            ids.append(_VOCAB.index(piece))
            i += len(piece) if pieces else 1
        return ids

    def decode(self, ids, **kwargs):
        return "".join(_VOCAB[i] for i in ids)

    def apply_chat_template(self, messages, **kwargs):
        return "".join(
            m["content"] + ("<eos>" if m["role"] == "assistant" else "")
            for m in messages
        )


TOK = TextTokenizer()


def _paged():
    return PagedCacheManager(
        block_size=4, max_blocks=16, initial_blocks=16, model_name="m"
    )


def _hold(paged, parent, tokens):
    """Put a block (full or tail) in the block map; return its hash."""
    block_hash = compute_block_hash(parent, tokens, model_name="m")
    block = paged.allocate_block()
    block.block_hash = block_hash
    block.token_count = len(tokens)
    paged.cached_block_hash_to_block.insert(block_hash, block)
    return block_hash


def _record_with_tail(paged, tail_text="ab<eos>"):
    """A full block "efgh" and, under it, a tail spelled a, b, <eos>."""
    parent = _hold(paged, None, TOK.encode("efgh"))
    tail_ids = [_VOCAB.index(c) for c in "ab"] + [EOS]
    tail_hash = _hold(paged, parent, tail_ids)
    paged.register_tail_block(parent, tail_hash, 3, text=tail_text, token_ids=tail_ids)
    return parent, tail_hash, tail_ids


class TestTailTextMatch:
    def test_prompt_spelling_the_tail_with_other_ids_is_spliced(self):
        paged = _paged()
        _, _, tail_ids = _record_with_tail(paged)
        prompt = TOK.encode("efghab<eos>cd")
        assert prompt[4:] == [_VOCAB.index("ab"), EOS, _VOCAB.index("cd")]
        assert paged.get_computed_blocks(prompt)[1] == 4

        spliced = paged.splice_tail_by_text(prompt, TOK)

        assert spliced == prompt[:4] + tail_ids + [_VOCAB.index("cd")]
        assert TOK.decode(spliced) == TOK.decode(prompt)
        assert paged.get_computed_blocks(spliced)[1] == 7

    def test_hash_match_still_wins_and_needs_no_splice(self):
        paged = _paged()
        _, _, tail_ids = _record_with_tail(paged)
        prompt = TOK.encode("efgh") + tail_ids + TOK.encode("cd")

        assert paged.splice_tail_by_text(prompt, TOK) is None
        assert paged.get_computed_blocks(prompt)[1] == 7

    def test_longest_text_wins(self):
        paged = _paged()
        parent, long_hash, _ = _record_with_tail(paged)
        short_hash = _hold(paged, parent, [_VOCAB.index("a")])
        paged.register_tail_block(
            parent, short_hash, 1, text="a", token_ids=[_VOCAB.index("a")]
        )
        prompt = TOK.encode("efghab<eos>cd")

        match = paged.match_tail_text(prompt, parent, 4, TOK)

        assert match is not None and match[:2] == (long_hash, 3)

    @pytest.mark.parametrize("text", ["efghabcd", "efghab", "efghba<eos>"])
    def test_text_that_does_not_begin_with_the_tail_misses(self, text):
        paged = _paged()
        _record_with_tail(paged)
        assert paged.splice_tail_by_text(TOK.encode(text), TOK) is None

    def test_tail_no_longer_held_misses(self):
        paged = _paged()
        _, tail_hash, _ = _record_with_tail(paged)
        block = paged.cached_block_hash_to_block.get_block(tail_hash)
        paged.cached_block_hash_to_block.pop(tail_hash, block.block_id)
        assert paged.splice_tail_by_text(TOK.encode("efghab<eos>cd"), TOK) is None

    def test_text_split_mid_character_is_not_kept(self):
        paged = _paged()
        parent, tail_hash, _ = _record_with_tail(paged, tail_text="�b<eos>")
        assert tail_hash in paged._tail_index[parent]
        assert tail_hash not in paged._tail_texts


def _ids(text):
    """Ids spelling ``text`` one character at a time."""
    return [_VOCAB.index(c) for c in text]


class TestTrailingWhitespace:
    """A client may strip trailing spaces and tabs from the lines it resends."""

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("agent? \nfoo\t", "agent?\nfoo"),
            ("a \t \n \nb", "a\n\nb"),
            ("a b\n", "a b\n"),
            ("", ""),
        ],
    )
    def test_normalize_strips_only_line_ends(self, raw, expected):
        assert normalize_tail_text(raw) == expected

    @pytest.mark.parametrize(
        "raw,length,offset",
        [
            ("ab  \ncd", 3, 5),
            ("a b", 2, 2),
            ("a  ", 1, 1),
            ("x \t\ny", 3, 5),
            ("abc", 0, 0),
        ],
    )
    def test_raw_offset_maps_a_normalized_length_back(self, raw, length, offset):
        assert _raw_tail_offset(raw, length) == offset
        assert normalize_tail_text(raw[offset:]) == normalize_tail_text(raw)[length:]

    def test_tail_with_a_trailing_space_matches_a_prompt_without_it(self):
        paged = _paged()
        parent = _hold(paged, None, TOK.encode("efgh"))
        tail_ids = _ids("a \n")
        tail_hash = _hold(paged, parent, tail_ids)
        paged.register_tail_block(parent, tail_hash, 3, text="a \n", token_ids=tail_ids)
        assert paged._tail_texts[tail_hash][0] == "a\n"
        prompt = TOK.encode("efgha\nbcd")

        spliced = paged.splice_tail_by_text(prompt, TOK)

        # The tail keeps the sampled space; the raw rest starts after "a\n".
        assert spliced == prompt[:4] + tail_ids + TOK.encode("bcd")
        assert paged.get_computed_blocks(spliced)[1] == 7

    def test_prompt_with_trailing_spaces_inside_the_tail_matches(self):
        paged = _paged()
        parent = _hold(paged, None, TOK.encode("efgh"))
        tail_ids = _ids("a\nb")
        tail_hash = _hold(paged, parent, tail_ids)
        paged.register_tail_block(parent, tail_hash, 3, text="a\nb", token_ids=tail_ids)
        prompt = TOK.encode("efgha \t\nbcd")

        match = paged.match_tail_text(prompt, parent, 4, TOK)

        assert match is not None and match[:2] == (tail_hash, 3)
        assert match[2] == prompt[:4] + tail_ids + TOK.encode("cd")

    def test_other_differences_still_miss(self):
        paged = _paged()
        parent = _hold(paged, None, TOK.encode("efgh"))
        tail_ids = _ids("a b")
        tail_hash = _hold(paged, parent, tail_ids)
        paged.register_tail_block(parent, tail_hash, 3, text="a b", token_ids=tail_ids)
        # A space inside a line is kept, so dropping it is a different text.
        assert paged.splice_tail_by_text(TOK.encode("efghabcd"), TOK) is None


class TestLongestTailWins:
    """Under one parent the longest tail wins, matched by hash or by text."""

    def _parent(self):
        paged = PagedCacheManager(
            block_size=8, max_blocks=16, initial_blocks=16, model_name="m"
        )
        full = TOK.encode("efghijkl")
        return paged, full, _hold(paged, None, full)

    def _tail(self, paged, parent, ids):
        tail_hash = _hold(paged, parent, ids)
        paged.register_tail_block(
            parent, tail_hash, len(ids), text=TOK.decode(ids), token_ids=ids
        )
        return tail_hash

    @staticmethod
    def _probe_paged(paged, token_ids):
        scheduler = SimpleNamespace(
            block_aware_cache=SimpleNamespace(block_size=8),
            paged_ssd_cache_manager=None,
            paged_cache_manager=paged,
            config=SimpleNamespace(paged_cache_block_size=8),
        )
        return _probe(scheduler, token_ids)

    def test_longer_tail_by_text_beats_shorter_tail_by_hash(self):
        paged, full, parent = self._parent()
        self._tail(paged, parent, _ids("ab"))
        long_ids = _ids("abcd") + [EOS]
        long_hash = self._tail(paged, parent, long_ids)
        # The older turn end matches by hash; the newer one only by text,
        # its "cd" re-tokenized as one id.
        prompt = full + _ids("ab") + [_VOCAB.index("cd"), EOS] + _ids("x")
        assert paged.get_computed_blocks(prompt)[1] == 10

        spliced = paged.splice_tail_by_text(prompt, TOK)

        assert spliced == full + long_ids + _ids("x")
        blocks, cached = paged.get_computed_blocks(spliced)
        assert cached == 13 and blocks[-1].block_hash == long_hash
        probe = self._probe_paged(paged, prompt)
        assert probe["live_tail_tokens"] == 5
        assert probe["live_hit_tokens"] == 13
        assert probe["live_prompt_tokens"] == 14

    def test_longer_tail_by_hash_beats_shorter_tail_by_text(self):
        paged, full, parent = self._parent()
        self._tail(paged, parent, [_VOCAB.index("ab")])
        long_ids = _ids("abcd") + [EOS]
        self._tail(paged, parent, long_ids)
        prompt = full + long_ids + _ids("x")

        assert paged.splice_tail_by_text(prompt, TOK) is None
        assert paged.get_computed_blocks(prompt)[1] == 13
        probe = self._probe_paged(paged, prompt)
        assert probe["live_tail_tokens"] == 5
        assert probe["live_hit_tokens"] == 13
        assert probe["live_prompt_tokens"] == 14


@pytest.mark.parametrize("raw_on_disk", [False, True])
def test_tail_text_persists_across_a_rescan(tmp_path, raw_on_disk):
    """The text and ids are saved with the tail's metadata and seed the index.

    The text is kept normalized; one saved raw by an older build is
    normalized when read back.
    """
    parent = compute_block_hash(None, [4, 5, 6, 7], model_name="test-model")
    tail_hash = compute_block_hash(parent, [0, 1, EOS], model_name="test-model")
    manager = PagedSSDCacheManager(cache_dir=tmp_path / "ssd", max_size_bytes=1024**3)
    keep_raw = patch(
        "omlx.cache.paged_ssd_cache.normalize_tail_text", side_effect=lambda t: t
    )
    try:
        if raw_on_disk:
            keep_raw.start()
        assert manager.save_block(
            block_hash=tail_hash,
            cache_data=[(mx.zeros((1, 2, 3, 8)), mx.zeros((1, 2, 3, 8)))],
            token_count=3,
            model_name="test-model",
            layer_cache_types=["KVCache"],
            parent_hash=parent,
            tail_terminal=True,
            tail_text="ab é \t\n<eos>",
            tail_token_ids=[0, 1, EOS],
        )
    finally:
        if raw_on_disk:
            keep_raw.stop()
        manager.close()

    reopened = PagedSSDCacheManager(cache_dir=tmp_path / "ssd", max_size_bytes=1024**3)
    try:
        meta = reopened.get_block_metadata(tail_hash)
        assert meta.tail_text == "ab é\n<eos>"
        assert list(meta.tail_token_ids) == [0, 1, EOS]
        assert [(h, t, list(i)) for h, t, i in reopened.iter_tail_texts()] == [
            (tail_hash, "ab é\n<eos>", [0, 1, EOS])
        ]
        paged = PagedCacheManager(
            block_size=4, max_blocks=16, initial_blocks=16, model_name="test-model"
        )
        BlockAwarePrefixCache(
            model=_tiny_hybrid_model(),
            paged_cache_manager=paged,
            paged_ssd_cache_manager=reopened,
        )
        text, ids = paged._tail_texts[tail_hash]
        assert text == "ab é\n<eos>" and list(ids) == [0, 1, EOS]
    finally:
        reopened.close()


def _tiny_llama_model():
    from mlx_lm.models.llama import Model, ModelArgs

    return Model(
        ModelArgs(
            model_type="llama",
            hidden_size=32,
            num_hidden_layers=2,
            intermediate_size=64,
            num_attention_heads=2,
            num_key_value_heads=1,
            rms_norm_eps=1e-5,
            vocab_size=32,
        )
    )


def _rig(tmp_path, kind="gdn", block_size=4):
    mx.random.seed(7)
    model = _tiny_hybrid_model() if kind == "gdn" else _tiny_llama_model()
    layer_types = ["ArraysCache", "KVCache"] if kind == "gdn" else ["KVCache"] * 2
    scheduler = Scheduler(
        model=model,
        tokenizer=TOK,
        config=SchedulerConfig(paged_cache_block_size=block_size),
    )
    paged = PagedCacheManager(
        block_size=block_size, max_blocks=32, initial_blocks=32, model_name="tiny-qwen"
    )
    ssd = PagedSSDCacheManager(
        cache_dir=tmp_path / "cache",
        max_size_bytes=16 * 1024**2,
        expected_model_name="tiny-qwen",
        expected_num_layers=2,
        expected_block_size=block_size,
        expected_layer_cache_types=layer_types,
    )
    boundary = BoundarySnapshotSSDStore(tmp_path / "cache")
    prefix = BlockAwarePrefixCache(
        model=model, paged_cache_manager=paged, paged_ssd_cache_manager=ssd
    )
    prefix.set_gdn_checkpoint_loader(boundary.load_file)
    scheduler.paged_cache_manager = paged
    scheduler.paged_ssd_cache_manager = ssd
    scheduler._boundary_snapshot_store = boundary
    scheduler.block_aware_cache = prefix
    scheduler._store_cache_executor = ThreadPoolExecutor(max_workers=1)
    return model, scheduler, paged, ssd, prefix


def _finish_turn(scheduler, prompt, sampled):
    """Generate ``sampled`` then the stop token, and store the finished request."""
    script = iter(sampled + [EOS])

    def sampler(logprobs):
        return mx.full(logprobs.shape[:-1], next(script, 0), dtype=mx.int32)

    generator = BatchGenerator(
        scheduler.model,
        max_tokens=16,
        stop_tokens=[[EOS]],
        sampler=sampler,
        stream=scheduler._stream,
    )
    scheduler.batch_generator = generator
    request = Request(
        request_id="turn",
        prompt=list(prompt),
        sampling_params=SamplingParams(max_tokens=16),
    )
    scheduler.add_request(request)
    scheduler.waiting.clear()
    scheduler.running[request.request_id] = request
    uid = generator.insert([list(prompt)])[0]
    scheduler.request_id_to_uid[request.request_id] = uid
    scheduler.uid_to_request_id[uid] = request.request_id
    finished = set()
    for _ in range(10):
        _, responses = generator.next()
        if responses:
            _, finished = scheduler._process_batch_responses(responses)
            if finished:
                break
    assert finished == {request.request_id}
    scheduler._cleanup_finished(finished)
    for future in list(scheduler._inflight_store_futures.values()):
        future.result(timeout=30)
    scheduler._drain_pending_async_removes()
    return request


def _probe(scheduler, token_ids):
    entry = SimpleNamespace(
        engine=SimpleNamespace(
            _tokenizer=TOK,
            _engine=SimpleNamespace(engine=SimpleNamespace(scheduler=scheduler)),
        )
    )
    pool = SimpleNamespace(_entries={"m": entry})
    with (
        patch.object(admin_routes, "_get_engine_pool", return_value=pool),
        patch("omlx.server._tokenize_chat_messages", AsyncMock(return_value=token_ids)),
    ):
        return asyncio.run(
            admin_routes.probe_cache(
                admin_routes.CacheProbeRequest(
                    model_id="m", messages=[{"role": "user", "content": "x"}]
                ),
                is_admin=True,
            )
        )


@pytest.mark.parametrize("kind", ["gdn", "kv"])
def test_turn_end_tail_serves_a_sibling_by_bytes(tmp_path, kind):
    """A finished turn is stored through its stop token; a later prompt that
    resends the turn with other ids for its bytes, then a different message,
    is spliced onto the stored tail, restores exactly, and the probe agrees.
    A hybrid cache stores the tail from a snapshot of the finished state, a
    cache whose layers all slice from the live cache."""
    model, scheduler, paged, ssd, prefix = _rig(tmp_path, kind)
    try:
        prompt = TOK.encode("efgha")
        request = _finish_turn(scheduler, prompt, [_VOCAB.index("b")])
        assert request.output_token_ids == [_VOCAB.index("b")]
        assert request.stop_token_id == EOS
        assert request.turn_end_tokens == 7
        turn = prompt + [_VOCAB.index("b"), EOS]
        parent = compute_block_hash(None, turn[:4], model_name="tiny-qwen")
        tail_hash = compute_block_hash(parent, turn[4:], model_name="tiny-qwen")
        assert paged._tail_index[parent][tail_hash] == 3
        assert paged._tail_texts[tail_hash][0] == "ab<eos>"
        assert ssd.get_block_metadata(tail_hash).tail_text == "ab<eos>"

        # The sibling re-tokenizes the turn: "ab" is one id, not a and b.
        sibling = TOK.encode("efghab<eos>cd")
        assert sibling[:4] == turn[:4] and sibling[4:6] != turn[4:6]
        spliced = turn + [_VOCAB.index("cd")]

        probe = _probe(scheduler, sibling)
        assert probe["live_hit_tokens"] == 7
        assert probe["live_tail_tokens"] == 3
        assert probe["live_prompt_tokens"] == 8

        next_request = Request(
            request_id="sibling",
            prompt=list(sibling),
            sampling_params=SamplingParams(max_tokens=1),
        )
        scheduler.add_request(next_request)
        scheduler._prepare_prefix_cache_for_request(next_request)
        assert next_request.prompt_token_ids == spliced
        assert next_request.num_prompt_tokens == 8
        assert next_request.cached_tokens == 7
        scheduler._release_paged_cache_for_request(next_request.request_id)

        table, remaining = prefix.fetch_cache("sibling", next_request.prompt_token_ids)
        assert table is not None and table.num_tokens == 7
        assert remaining == [_VOCAB.index("cd")]
        restored = prefix.reconstruct_cache(table)
        assert restored is not None
        with mx.stream(scheduler._stream):
            warm = model(mx.array([remaining]), cache=restored)[:, -1]
            cold = model(mx.array([spliced]), cache=model.make_cache())[:, -1]
            mx.eval(warm, cold)
        assert mx.allclose(warm, cold, atol=1e-4)
        prefix.release_cache("sibling")

        # The record's own ids still match by hash, with nothing to splice.
        assert paged.splice_tail_by_text(spliced, TOK) is None
        assert _probe(scheduler, spliced)["live_hit_tokens"] == 7
    finally:
        scheduler.shutdown()


def test_sampled_trailing_space_matches_the_stripped_resend(tmp_path):
    """The model ends a line with a space; the client resends the line without
    it. The tail still serves the next prompt and keeps what the model wrote."""
    model, scheduler, paged, ssd, prefix = _rig(tmp_path, block_size=8)
    try:
        prompt = TOK.encode("efghijkl")
        request = _finish_turn(scheduler, prompt, _ids("a \n"))
        turn = prompt + _ids("a \n") + [EOS]
        assert request.turn_end_tokens == len(turn) == 12
        parent = compute_block_hash(None, turn[:8], model_name="tiny-qwen")
        tail_hash = compute_block_hash(parent, turn[8:], model_name="tiny-qwen")
        assert paged._tail_texts[tail_hash][0] == "a\n<eos>"
        assert ssd.get_block_metadata(tail_hash).tail_text == "a\n<eos>"

        resent = TOK.encode("efghijkla\n<eos>cd")
        spliced = turn + TOK.encode("cd")
        assert _probe(scheduler, resent)["live_hit_tokens"] == 12

        next_request = Request(
            request_id="resent",
            prompt=list(resent),
            sampling_params=SamplingParams(max_tokens=1),
        )
        scheduler.add_request(next_request)
        scheduler._prepare_prefix_cache_for_request(next_request)
        assert next_request.prompt_token_ids == spliced
        assert next_request.cached_tokens == 12
        scheduler._release_paged_cache_for_request(next_request.request_id)

        table, remaining = prefix.fetch_cache("resent", spliced)
        assert table is not None and table.num_tokens == 12
        restored = prefix.reconstruct_cache(table)
        with mx.stream(scheduler._stream):
            warm = model(mx.array([remaining]), cache=restored)[:, -1]
            cold = model(mx.array([spliced]), cache=model.make_cache())[:, -1]
            mx.eval(warm, cold)
        assert mx.allclose(warm, cold, atol=1e-4)
        prefix.release_cache("resent")
    finally:
        scheduler.shutdown()


def test_turn_end_excludes_a_stop_token_the_template_does_not_render(tmp_path):
    """A hybrid cache cannot cut back past the stop token, so no tail is kept."""
    model, scheduler, paged, ssd, prefix = _rig(tmp_path)
    tokenizer = TextTokenizer()
    tokenizer.apply_chat_template = lambda messages, **kw: "".join(
        m["content"] for m in messages
    )
    scheduler.tokenizer = tokenizer
    try:
        request = _finish_turn(scheduler, TOK.encode("efgha"), [_VOCAB.index("b")])
        assert request.stop_token_id == EOS
        assert request.turn_end_tokens == 0
        assert not any(paged._tail_texts)
    finally:
        scheduler.shutdown()


@pytest.mark.parametrize(
    "headers,expected",
    [
        ({"x-omlx-cache-pin": "on"}, {"pin_cache": True}),
        ({"x-cache-pin": "on"}, {"pin_cache": True}),
        ({"x-cache-pin": " ON "}, {"pin_cache": True}),
        ({"x-cache-pin": "off"}, {}),
        ({}, {}),
    ],
)
def test_cache_pin_header_alias(headers, expected):
    from omlx.server import _cache_store_kwargs

    request = SimpleNamespace(headers=Headers(headers))
    assert _cache_store_kwargs(request) == expected

# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Issue #784: a worker deregistered by a transient Redis outage rejoins on heartbeat.

When Redis is unreachable longer than the worker-key TTL, the key expires,
clean_worker_registry SREMs the worker from rq:workers, and on reconnect the
heartbeat recreates only a partial hash - historically the live worker kept taking
jobs while invisible to the dashboard. These tests drive the real worker classes
against an in-memory Redis stand-in and assert the heartbeat re-registers.

Main Features:
* Both the forking Worker and the Windows SimpleWorker carry the mixin and rejoin.
* A heartbeat re-adds the worker to rq:workers and rq:workers:<queue> and restores
  the identity hash (birth/hostname/queues) when the key came back partial.
* Re-registration is idempotent and does not rewrite the hash during normal beats.
"""

import types

import pytest
from rq.utils import now

import rq_heartbeat_worker as rhw


class FakeRedis:
    def __init__(self):
        self.hashes = {}
        self.sets = {}
        self.ttls = {}
        self.connection_pool = types.SimpleNamespace(connection_kwargs={})

    def hset(self, name, key=None, value=None, mapping=None):
        h = self.hashes.setdefault(name, {})
        added = 0
        if mapping:
            for k, v in mapping.items():
                if k not in h:
                    added += 1
                h[k] = '' if v is None else str(v)
        if key is not None:
            if key not in h:
                added += 1
            h[key] = '' if value is None else str(value)
        return added

    def hexists(self, name, key):
        return key in self.hashes.get(name, {})

    def sadd(self, name, *members):
        s = self.sets.setdefault(name, set())
        before = len(s)
        s.update(members)
        return len(s) - before

    def srem(self, name, *members):
        s = self.sets.setdefault(name, set())
        before = len(s)
        for m in members:
            s.discard(m)
        return before - len(s)

    def smembers(self, name):
        return set(self.sets.get(name, set()))

    def scard(self, name):
        return len(self.sets.get(name, set()))

    def exists(self, *keys):
        return sum(1 for k in keys if k in self.hashes or k in self.sets)

    def expire(self, name, ttl):
        self.ttls[name] = ttl
        return 1

    def delete(self, *keys):
        removed = 0
        for k in keys:
            if k in self.hashes:
                del self.hashes[k]
                removed += 1
            if k in self.sets:
                del self.sets[k]
                removed += 1
        return removed


def _make_worker(worker_cls):
    fake = FakeRedis()
    worker = worker_cls(
        ['default'],
        connection=fake,
        prepare_for_work=False,
        worker_ttl=120,
        job_monitoring_interval=30,
        name='w784',
    )
    worker.birth_date = now()
    worker.last_heartbeat = now()
    worker.hostname = 'testhost'
    worker.pid = 4321
    worker.ip_address = '10.0.0.9'
    return worker, fake


@pytest.mark.parametrize('worker_cls', [rhw.ReregisteringWorker, rhw.HeartbeatSimpleWorker])
def test_partial_key_after_outage_rejoins_registry_on_heartbeat(worker_cls):
    worker, fake = _make_worker(worker_cls)
    fake.hashes[worker.key] = {'last_heartbeat': 'stale', 'successful_job_count': '400'}
    fake.sets['rq:workers'] = set()

    worker.heartbeat()

    assert worker.key in fake.smembers('rq:workers')
    assert worker.key in fake.smembers('rq:workers:default')


@pytest.mark.parametrize('worker_cls', [rhw.ReregisteringWorker, rhw.HeartbeatSimpleWorker])
def test_partial_key_identity_hash_is_restored_on_heartbeat(worker_cls):
    worker, fake = _make_worker(worker_cls)
    fake.hashes[worker.key] = {'last_heartbeat': 'stale', 'successful_job_count': '400'}
    fake.sets['rq:workers'] = set()

    worker.heartbeat()

    restored = fake.hashes[worker.key]
    assert restored.get('birth')
    assert restored.get('hostname') == 'testhost'
    assert restored.get('queues') == 'default'


@pytest.mark.parametrize('worker_cls', [rhw.ReregisteringWorker, rhw.HeartbeatSimpleWorker])
def test_heartbeat_reregistration_is_idempotent_when_already_registered(worker_cls):
    worker, fake = _make_worker(worker_cls)
    fake.hashes[worker.key] = {'birth': 'ORIGINAL', 'last_heartbeat': 'stale'}
    fake.sets['rq:workers'] = {worker.key}

    worker.heartbeat()

    assert fake.scard('rq:workers') == 1
    assert fake.hashes[worker.key]['birth'] == 'ORIGINAL'


@pytest.mark.parametrize('worker_cls', [rhw.ReregisteringWorker, rhw.HeartbeatSimpleWorker])
def test_pipeline_heartbeat_queues_reregistration_without_restore(worker_cls):
    worker, fake = _make_worker(worker_cls)
    fake.hashes[worker.key] = {'last_heartbeat': 'stale'}
    fake.sets['rq:workers'] = set()

    worker.heartbeat(pipeline=fake)

    assert worker.key in fake.smembers('rq:workers')
    assert 'birth' not in fake.hashes[worker.key]

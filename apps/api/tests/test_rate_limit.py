from app.core.rate_limit import RateLimiter


class FakePipeline:
    def __init__(self, redis_store):
        self.redis_store = redis_store
        self.operations = []

    def incr(self, key):
        self.operations.append(("incr", key))
        return self

    def expire(self, key, seconds):
        self.operations.append(("expire", key, seconds))
        return self

    def execute(self):
        results = []

        for operation in self.operations:
            if operation[0] == "incr":
                key = operation[1]
                self.redis_store[key] = self.redis_store.get(key, 0) + 1
                results.append(self.redis_store[key])

            elif operation[0] == "expire":
                results.append(True)

        return results


class FakeRedis:
    def __init__(self):
        self.store = {}

    def pipeline(self):
        return FakePipeline(self.store)

    def close(self):
        pass


def test_rate_limiter_allows_requests_under_limit(monkeypatch):
    fake_redis = FakeRedis()

    monkeypatch.setattr(
        "app.core.rate_limit.settings.rate_limit_enabled",
        True,
    )
    monkeypatch.setattr(
        "app.core.rate_limit.settings.rate_limit_requests",
        2,
    )

    limiter = RateLimiter()
    limiter.client = fake_redis

    try:
        allowed, retry_after = limiter.check("test-client")

        assert allowed is True
        assert retry_after == 0
    finally:
        limiter.close()


def test_rate_limiter_rejects_requests_over_limit(monkeypatch):
    fake_redis = FakeRedis()

    monkeypatch.setattr(
        "app.core.rate_limit.settings.rate_limit_enabled",
        True,
    )
    monkeypatch.setattr(
        "app.core.rate_limit.settings.rate_limit_requests",
        2,
    )

    limiter = RateLimiter()
    limiter.client = fake_redis

    try:
        first_allowed, _ = limiter.check("test-client")
        second_allowed, _ = limiter.check("test-client")
        third_allowed, retry_after = limiter.check("test-client")

        assert first_allowed is True
        assert second_allowed is True
        assert third_allowed is False
        assert retry_after >= 1
    finally:
        limiter.close()


def test_rate_limiter_disabled(monkeypatch):
    monkeypatch.setattr(
        "app.core.rate_limit.settings.rate_limit_enabled",
        False,
    )

    limiter = RateLimiter()

    try:
        allowed, retry_after = limiter.check("test-client")

        assert allowed is True
        assert retry_after == 0
    finally:
        limiter.close()

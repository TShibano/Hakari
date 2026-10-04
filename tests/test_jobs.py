import pytest

from hakari.batch import Tag, read_csv
from hakari.decision import ChoiceProbability, DecisionError, DecisionResult
from hakari.jobs import Job, JobStore, run_job

TAGS = [Tag("価格", "価格への言及"), Tag("接客", "スタッフの対応への言及")]


class FakeClient:
    def __init__(self, fail_on=None, on_decide=None):
        self.fail_on = fail_on
        self.on_decide = on_decide
        self.calls = 0

    def decide(self, req):
        self.calls += 1
        if self.on_decide:
            self.on_decide(req)
        if self.fail_on and self.fail_on in req.context:
            raise DecisionError("Ollama が落ちた")
        p = 0.9 if req.field in req.context else 0.2
        probs = sorted(
            [ChoiceProbability("A", "true", p), ChoiceProbability("B", "false", 1 - p)],
            key=lambda x: x.probability,
            reverse=True,
        )
        return DecisionResult(answer=probs[0].label, probabilities=probs)


def make_job(csv_text="id,感想\n1,価格が高い\n2,接客が良い\n3,\n"):
    return Job(table=read_csv(csv_text.encode()), column=1, tags=TAGS)


def test_new_job_is_pending():
    job = make_job()
    assert job.status == "pending"
    assert job.done == 0
    assert job.total == 3
    assert len(job.id) >= 16


def test_run_job_completes_and_summarizes():
    job = make_job()
    run_job(job, FakeClient())

    assert job.status == "done"
    assert job.done == 3
    assert job.error is None
    assert job.summary() == {"価格": 1, "接客": 1}


def test_run_job_updates_progress_while_running():
    job = make_job()
    seen = []
    run_job(job, FakeClient(on_decide=lambda req: seen.append((job.status, job.done))))

    assert seen[0] == ("running", 0)
    assert ("running", 1) in seen


def test_run_job_keeps_partial_results_on_failure():
    job = make_job("id,感想\n1,価格\n2,壊れる\n3,接客\n")
    run_job(job, FakeClient(fail_on="壊れる"))

    assert job.status == "failed"
    assert job.error == "Ollama が落ちた"
    assert job.done == 1
    assert job.summary() == {"価格": 1, "接客": 0}


def test_cancel_stops_after_current_row():
    job = make_job("id,感想\n1,a\n2,b\n3,c\n")
    client = FakeClient(on_decide=lambda req: job.cancel())
    run_job(job, client)

    assert job.status == "cancelled"
    assert job.done == 1
    assert client.calls == len(TAGS)


def test_result_csv_contains_processed_rows():
    job = make_job()
    run_job(job, FakeClient())
    text = job.result_csv().decode("utf-8-sig")
    assert text.splitlines()[0] == "id,感想,価格,価格_確率,接客,接客_確率"
    assert len(text.splitlines()) == 4


def test_snapshot_is_json_friendly():
    job = make_job()
    run_job(job, FakeClient())
    assert job.snapshot() == {
        "id": job.id,
        "status": "done",
        "done": 3,
        "total": 3,
        "error": None,
        "summary": {"価格": 1, "接客": 1},
    }


# --- JobStore ---


def test_store_starts_job_with_runner():
    started = []
    store = JobStore(client=FakeClient(), runner=lambda fn: started.append(fn))
    job = store.start(make_job())

    assert store.get(job.id) is job
    assert job.status == "pending"
    started[0]()
    assert job.status == "done"


def test_store_get_unknown_returns_none():
    store = JobStore(client=FakeClient(), runner=lambda fn: fn())
    assert store.get("nope") is None


def test_store_default_runner_runs_in_background_thread():
    store = JobStore(client=FakeClient())
    job = store.start(make_job())
    assert job.wait(timeout=5)
    assert job.status == "done"


def test_store_drops_oldest_finished_jobs_beyond_limit():
    store = JobStore(client=FakeClient(), runner=lambda fn: fn(), max_jobs=2)
    a = store.start(make_job())
    b = store.start(make_job())
    c = store.start(make_job())
    assert store.get(a.id) is None
    assert store.get(b.id) is b
    assert store.get(c.id) is c

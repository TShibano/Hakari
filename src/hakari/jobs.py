"""一括タグ付けをバックグラウンドで実行し，進捗と結果を保持する．"""

import secrets
import threading
from collections.abc import Callable
from dataclasses import dataclass, field

from hakari.batch import RowResult, Table, Tag, iter_tag_rows, summarize, write_csv
from hakari.decision import DecisionClient, DecisionError

FINISHED = ("done", "failed", "cancelled")


@dataclass
class Job:
    table: Table
    column: int
    tags: list[Tag]
    id: str = field(default_factory=lambda: secrets.token_urlsafe(16))
    status: str = "pending"
    error: str | None = None
    results: list[RowResult] = field(default_factory=list)
    _cancel: threading.Event = field(default_factory=threading.Event, repr=False)
    _finished: threading.Event = field(default_factory=threading.Event, repr=False)

    @property
    def total(self) -> int:
        return len(self.table.rows)

    @property
    def done(self) -> int:
        return len(self.results)

    def cancel(self) -> None:
        self._cancel.set()

    def wait(self, timeout: float | None = None) -> bool:
        return self._finished.wait(timeout)

    def summary(self) -> dict[str, int]:
        return summarize(self.tags, list(self.results))

    def result_csv(self) -> bytes:
        return write_csv(self.table, self.tags, list(self.results))

    def snapshot(self) -> dict:
        return {
            "id": self.id,
            "status": self.status,
            "done": self.done,
            "total": self.total,
            "error": self.error,
            "summary": self.summary(),
        }


def run_job(job: Job, client: DecisionClient) -> None:
    job.status = "running"
    try:
        for row in iter_tag_rows(client, job.table, job.column, job.tags):
            job.results.append(row)
            if job._cancel.is_set() and job.done < job.total:
                job.status = "cancelled"
                return
        job.status = "done"
    except DecisionError as e:
        job.status = "failed"
        job.error = str(e)
    except Exception as e:  # スレッド内の想定外の例外で status が running のまま残らないようにする．
        job.status = "failed"
        job.error = f"想定外のエラー: {e}"
    finally:
        job._finished.set()


def _thread_runner(fn: Callable[[], None]) -> None:
    threading.Thread(target=fn, daemon=True).start()


class JobStore:
    """ジョブをメモリ上に保持する．再起動で消えるが，単一利用者のローカル用途なので割り切る．"""

    def __init__(
        self,
        client: DecisionClient,
        runner: Callable[[Callable[[], None]], None] = _thread_runner,
        max_jobs: int = 20,
    ):
        self.client = client
        self.runner = runner
        self.max_jobs = max_jobs
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def start(self, job: Job) -> Job:
        with self._lock:
            self._evict()
            self._jobs[job.id] = job
        self.runner(lambda: run_job(job, self.client))
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def _evict(self) -> None:
        # 実行中のジョブは消さず，終わったものから古い順に消す．
        finished = [j for j in self._jobs.values() if j.status in FINISHED]
        while len(self._jobs) >= self.max_jobs and finished:
            del self._jobs[finished.pop(0).id]

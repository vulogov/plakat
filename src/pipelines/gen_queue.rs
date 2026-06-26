//! `GenQueue` — the ordered, bounded generation queue + single worker (RFC TUI-1
//! §14). One generation runs at a time (Metal device exclusivity), so requests
//! queue FIFO up to [`MAX_DEPTH`] and a single background worker drains them. The
//! UI enqueues a self-contained job (it captured its own [`GenMessage`] sender via
//! a [`ChannelHook`]) and keeps the matching [`CancelFlag`] so it can cancel.
//!
//! Pure `std` (no `ratatui`): the bounded-FIFO + cancel-registry core is a plain
//! testable struct, and [`GenWorker`] is a thin condvar-driven thread over it.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};

use super::gen_channel::CancelFlag;

/// Maximum queued (not-yet-running) requests. A 6th enqueue is rejected.
pub const MAX_DEPTH: usize = 5;

/// A unit of work for the queue. `run` is self-contained — it already captured the
/// pipeline, request, channel sender, and a [`ChannelHook`] built over `cancel`.
pub struct GenJob {
    pub id: u64,
    pub label: String,
    pub cancel: CancelFlag,
    run: Box<dyn FnOnce() + Send>,
}

impl GenJob {
    pub fn new(label: impl Into<String>, cancel: CancelFlag, run: impl FnOnce() + Send + 'static) -> Self {
        // id is assigned by the queue on enqueue; 0 until then.
        Self { id: 0, label: label.into(), cancel, run: Box::new(run) }
    }
}

/// Rejected because the queue was full.
#[derive(Debug, PartialEq, Eq)]
pub struct QueueFull;

/// The bounded FIFO + the currently-running job's cancel handle. Not thread-aware
/// on its own — [`GenWorker`] wraps it for background draining.
pub struct GenQueue {
    queue: VecDeque<GenJob>,
    running: Option<(u64, CancelFlag)>,
    capacity: usize,
    next_id: u64,
}

impl GenQueue {
    pub fn new() -> Self {
        Self::with_capacity(MAX_DEPTH)
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self { queue: VecDeque::new(), running: None, capacity, next_id: 1 }
    }

    /// Enqueue, assigning + returning a stable job id. `Err(QueueFull)` when the
    /// queue already holds `capacity` waiting jobs (the running one doesn't count).
    pub fn enqueue(&mut self, mut job: GenJob) -> Result<u64, QueueFull> {
        if self.queue.len() >= self.capacity {
            return Err(QueueFull);
        }
        let id = self.next_id;
        self.next_id += 1;
        job.id = id;
        self.queue.push_back(job);
        Ok(id)
    }

    /// Take the next waiting job (FIFO) and record it as running.
    pub fn take_next(&mut self) -> Option<GenJob> {
        let job = self.queue.pop_front()?;
        self.running = Some((job.id, job.cancel.clone()));
        Some(job)
    }

    /// Mark the running job finished.
    pub fn finish_running(&mut self) {
        self.running = None;
    }

    /// Cancel the running job, if its id matches (or unconditionally with `None`).
    /// Returns whether a running job was signalled.
    pub fn cancel_running(&mut self) -> bool {
        match &self.running {
            Some((_, flag)) => {
                flag.cancel();
                true
            }
            None => false,
        }
    }

    /// Cancel a job by id whether it's running (signal it) or still queued (drop
    /// it). Returns whether anything matched.
    pub fn cancel(&mut self, id: u64) -> bool {
        if let Some((rid, flag)) = &self.running {
            if *rid == id {
                flag.cancel();
                return true;
            }
        }
        if let Some(pos) = self.queue.iter().position(|j| j.id == id) {
            self.queue.remove(pos);
            return true;
        }
        false
    }

    pub fn running_id(&self) -> Option<u64> {
        self.running.as_ref().map(|(id, _)| *id)
    }
    pub fn waiting(&self) -> usize {
        self.queue.len()
    }
    pub fn is_full(&self) -> bool {
        self.queue.len() >= self.capacity
    }
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty() && self.running.is_none()
    }
}

impl Default for GenQueue {
    fn default() -> Self {
        Self::new()
    }
}

/// A single background thread that drains a shared [`GenQueue`] FIFO, running one
/// job at a time. The UI enqueues via [`GenWorker::enqueue`] (which wakes the
/// worker) and drains the per-job message channels on its own tick.
pub struct GenWorker {
    shared: Arc<(Mutex<GenQueue>, Condvar)>,
    shutdown: Arc<AtomicBool>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl GenWorker {
    /// Spawn the worker thread. It blocks on the condvar when idle.
    pub fn spawn() -> Self {
        let shared = Arc::new((Mutex::new(GenQueue::new()), Condvar::new()));
        let shutdown = Arc::new(AtomicBool::new(false));
        let handle = {
            let shared = Arc::clone(&shared);
            let shutdown = Arc::clone(&shutdown);
            std::thread::Builder::new()
                .name("plakat-gen-worker".into())
                .spawn(move || worker_loop(&shared, &shutdown))
                .expect("spawn gen worker")
        };
        Self { shared, shutdown, handle: Some(handle) }
    }

    /// Enqueue a job and wake the worker. `Err(QueueFull)` if the queue is full.
    pub fn enqueue(&self, job: GenJob) -> Result<u64, QueueFull> {
        let (lock, cvar) = &*self.shared;
        let id = lock.lock().unwrap().enqueue(job)?;
        cvar.notify_one();
        Ok(id)
    }

    pub fn cancel_running(&self) -> bool {
        self.shared.0.lock().unwrap().cancel_running()
    }
    pub fn cancel(&self, id: u64) -> bool {
        self.shared.0.lock().unwrap().cancel(id)
    }
    pub fn waiting(&self) -> usize {
        self.shared.0.lock().unwrap().waiting()
    }
    pub fn running_id(&self) -> Option<u64> {
        self.shared.0.lock().unwrap().running_id()
    }
}

impl Drop for GenWorker {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Relaxed);
        self.shared.1.notify_all();
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

fn worker_loop(shared: &Arc<(Mutex<GenQueue>, Condvar)>, shutdown: &Arc<AtomicBool>) {
    let (lock, cvar) = &**shared;
    loop {
        if shutdown.load(Ordering::Relaxed) {
            return;
        }
        // Pop a job (or wait for one), holding the lock only to take it.
        let job = {
            let mut q = lock.lock().unwrap();
            while q.is_empty() && !shutdown.load(Ordering::Relaxed) {
                q = cvar.wait(q).unwrap();
            }
            if shutdown.load(Ordering::Relaxed) {
                return;
            }
            q.take_next()
        };
        if let Some(job) = job {
            (job.run)(); // runs outside the lock — generation can take minutes
            lock.lock().unwrap().finish_running();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn job(label: &str, run: impl FnOnce() + Send + 'static) -> GenJob {
        GenJob::new(label, CancelFlag::new(), run)
    }

    #[test]
    fn enqueue_assigns_increasing_ids() {
        let mut q = GenQueue::new();
        assert_eq!(q.enqueue(job("a", || {})), Ok(1));
        assert_eq!(q.enqueue(job("b", || {})), Ok(2));
        assert_eq!(q.waiting(), 2);
    }

    #[test]
    fn rejects_when_full() {
        let mut q = GenQueue::with_capacity(2);
        q.enqueue(job("a", || {})).unwrap();
        q.enqueue(job("b", || {})).unwrap();
        assert_eq!(q.enqueue(job("c", || {})), Err(QueueFull));
    }

    #[test]
    fn take_next_is_fifo_and_marks_running() {
        let mut q = GenQueue::new();
        let id1 = q.enqueue(job("a", || {})).unwrap();
        q.enqueue(job("b", || {})).unwrap();
        let taken = q.take_next().unwrap();
        assert_eq!(taken.id, id1);
        assert_eq!(q.running_id(), Some(id1));
        assert_eq!(q.waiting(), 1);
    }

    #[test]
    fn cancel_running_sets_the_flag() {
        let mut q = GenQueue::new();
        let cancel = CancelFlag::new();
        q.enqueue(GenJob::new("a", cancel.clone(), || {})).unwrap();
        q.take_next();
        assert!(q.cancel_running());
        assert!(cancel.is_cancelled());
    }

    #[test]
    fn cancel_queued_removes_it() {
        let mut q = GenQueue::new();
        q.enqueue(job("a", || {})).unwrap();
        let id2 = q.enqueue(job("b", || {})).unwrap();
        assert!(q.cancel(id2));
        assert_eq!(q.waiting(), 1);
        assert!(!q.cancel(999)); // unknown id
    }

    #[test]
    fn worker_runs_jobs_in_order() {
        use std::sync::mpsc::channel;
        let worker = GenWorker::spawn();
        let (tx, rx) = channel();
        for n in 0..3 {
            let tx = tx.clone();
            worker.enqueue(job(&format!("j{n}"), move || tx.send(n).unwrap())).unwrap();
        }
        drop(tx);
        let got: Vec<i32> = rx.iter().collect(); // worker drops senders as it finishes
        assert_eq!(got, vec![0, 1, 2]);
        // worker shuts down cleanly on drop
    }
}

//! Single-instance guard — refuse to start a second heavy plakat run on the
//! same host. On unified-memory machines (Apple Silicon) two concurrent
//! model/training runs share the SAME memory pool and thrash each other into
//! swap (or trip the [`crate::memwatch`] OOM guard); the symptom is a run that
//! "randomly" OOMs because an earlier instance is still resident.
//!
//! Scans the OS process list for OTHER processes whose executable is `plakat`
//! — across users and install paths (matched by both the process name and the
//! executable's file name). Disabled with `--enable-multiple-instances` (or
//! `PLAKAT_ALLOW_MULTIPLE_INSTANCES=1`). When it fires it reports exactly which
//! instance it found (pid + uid + path + command) so the user can act.

use anyhow::{Result, bail};

/// One other plakat process found on the host.
#[derive(Debug, Clone)]
pub struct OtherInstance {
    pub pid: u32,
    /// Numeric user id (resolving to a name needs extra privileges/features).
    pub uid: Option<String>,
    /// Absolute executable path, when the OS exposes it.
    pub exe: Option<String>,
    /// Full command line, when readable (often restricted for other users).
    pub cmd: String,
}

/// Find every OTHER process on the host that is a `plakat` executable.
pub fn find_other_plakat_instances() -> Vec<OtherInstance> {
    use sysinfo::{ProcessRefreshKind, ProcessesToUpdate, System};

    let self_pid = std::process::id();
    let mut sys = System::new();
    // Refresh ALL processes with full detail (exe + cmd + user) so cross-user /
    // cross-path instances are visible where the OS permits.
    sys.refresh_processes_specifics(
        ProcessesToUpdate::All,
        true,
        ProcessRefreshKind::everything(),
    );

    let mut found = Vec::new();
    for (pid, proc) in sys.processes() {
        let p = pid.as_u32();
        if p == self_pid {
            continue;
        }
        let name = proc.name().to_string_lossy().to_ascii_lowercase();
        let exe_base = proc
            .exe()
            .and_then(|e| e.file_name())
            .map(|f| f.to_string_lossy().to_ascii_lowercase());
        // Match on the process name OR the executable's basename so a renamed
        // launcher (e.g. `./target/release/plakat`) and an installed `plakat`
        // both register, regardless of user or install path.
        let is_plakat = name == "plakat" || exe_base.as_deref() == Some("plakat");
        if !is_plakat {
            continue;
        }
        let cmd = proc
            .cmd()
            .iter()
            .map(|s| s.to_string_lossy())
            .collect::<Vec<_>>()
            .join(" ");
        found.push(OtherInstance {
            pid: p,
            uid: proc.user_id().map(|u| u.to_string()),
            exe: proc.exe().map(|e| e.display().to_string()),
            cmd: if cmd.is_empty() { name } else { cmd },
        });
    }
    found.sort_by_key(|o| o.pid);
    found
}

/// Bail if another plakat instance is already running, unless overridden.
pub fn enforce_single_instance(allow_multiple: bool) -> Result<()> {
    if allow_multiple || std::env::var_os("PLAKAT_ALLOW_MULTIPLE_INSTANCES").is_some() {
        return Ok(());
    }
    let others = find_other_plakat_instances();
    if others.is_empty() {
        return Ok(());
    }
    let mut msg = String::from(
        "another plakat instance is already running on this host — refusing to start a \
         second heavy run (they share unified memory and thrash). Found:\n",
    );
    for o in &others {
        let uid = o.uid.as_deref().map(|u| format!(" uid {u}")).unwrap_or_default();
        let path = o.exe.as_deref().unwrap_or("<path hidden>");
        msg.push_str(&format!("  • pid {}{uid}  {path}\n", o.pid));
        if !o.cmd.is_empty() {
            // Trim a very long command line so the message stays readable.
            let c: String = o.cmd.chars().take(160).collect();
            msg.push_str(&format!("      {c}\n"));
        }
    }
    msg.push_str(
        "\n  Wait for it to finish, or re-run with --enable-multiple-instances \
         (env PLAKAT_ALLOW_MULTIPLE_INSTANCES=1) if you really want concurrent runs.",
    );
    bail!("{msg}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allow_flag_short_circuits_without_scanning() {
        // With the override on, it must return Ok regardless of host state.
        assert!(enforce_single_instance(true).is_ok());
    }

    #[test]
    fn scan_excludes_self_and_returns_a_list() {
        // The scan must never report THIS process (the test runner is not
        // named plakat, but self-exclusion must hold regardless). Smoke: it
        // runs without panicking and never includes our own pid.
        let self_pid = std::process::id();
        let found = find_other_plakat_instances();
        assert!(found.iter().all(|o| o.pid != self_pid));
    }
}

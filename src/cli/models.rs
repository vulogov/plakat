use anyhow::Result;
use clap::Subcommand;

#[derive(Subcommand, Debug)]
pub enum ModelsCmd {
    /// Free-text search of HuggingFace Hub.
    Search {
        query: String,
        #[arg(long, default_value_t = 10)]
        limit: usize,
    },
    /// Recommended text-to-image models from HuggingFace.
    Recommend {
        /// Optional search filter, e.g. "sdxl", "anime", "realistic".
        #[arg(long)]
        query: Option<String>,
        /// Sort criterion: downloads | likes | trending | recent.
        #[arg(long, default_value = "downloads")]
        sort: String,
        #[arg(long, default_value_t = 15)]
        limit: usize,
    },
    /// Report file sizes for a repo and the subset plakat would download.
    Size { repo: String },
    /// Pull common SD weight files for a repo into the local cache.
    Pull { repo: String },
    /// List cached models.
    Ls,
    /// Remove one or more models from the cache (asks before each unless --yes).
    Rm {
        /// One or more repo ids or aliases.
        #[arg(required = true)]
        repos: Vec<String>,
        /// Skip the confirmation prompt.
        #[arg(long, short = 'y')]
        yes: bool,
    },
    /// List every `--model` alias plakat recognises, grouped by family.
    ///
    /// v0.20 #4: enumerates the static alias table. Use `--family
    /// flux` to filter to a single family, `--repo` to print HF
    /// repo ids only (suitable for piping into `xargs plakat
    /// models pull`), or `--gated` to filter to HF_TOKEN-gated
    /// repos.
    Aliases {
        /// Filter rows to one family (case-insensitive substring
        /// match against the family heading, e.g. "flux", "sdxl",
        /// "sd 3").
        #[arg(long)]
        family: Option<String>,
        /// Print canonical HF repo ids only (one per line, no
        /// headings) — handy for piping into other commands.
        #[arg(long)]
        repo: bool,
        /// Only list repos that require an HF_TOKEN (gated).
        #[arg(long)]
        gated: bool,
    },
}

pub async fn run(cmd: ModelsCmd) -> Result<()> {
    match cmd {
        ModelsCmd::Search { query, limit } => crate::hf::search::print_search(&query, limit).await,
        ModelsCmd::Recommend {
            query,
            sort,
            limit,
        } => crate::hf::search::print_recommend(query.as_deref(), &sort, limit).await,
        ModelsCmd::Size { repo } => crate::hf::info::print_size(&repo).await,
        ModelsCmd::Pull { repo } => crate::hf::download::pull_all(&repo).await,
        ModelsCmd::Ls => crate::hf::cache::list(),
        ModelsCmd::Rm { repos, yes } => crate::hf::cache::remove_many(&repos, yes),
        ModelsCmd::Aliases {
            family,
            repo,
            gated,
        } => print_aliases(family.as_deref(), repo, gated),
    }
}

/// v0.20 #4: render the alias table to stdout. Two layouts:
///
/// * Default (grouped) — headings per family, then `alias1, alias2, …
///   → repo` with note + gated marker. Easy to scan for "what can I
///   pass to --model?"
/// * `--repo` — bare repo ids, one per line, no headings. Pipes
///   cleanly into `xargs plakat models pull` for a bulk warm-up.
fn print_aliases(family: Option<&str>, repo_only: bool, gated_only: bool) -> Result<()> {
    let needle = family.map(str::to_ascii_lowercase);
    let entries: Vec<&crate::hf::AliasEntry> = crate::hf::ALIAS_TABLE
        .iter()
        .filter(|e| {
            needle
                .as_deref()
                .map(|n| e.family.to_ascii_lowercase().contains(n))
                .unwrap_or(true)
        })
        .filter(|e| !gated_only || e.gated)
        .collect();

    if entries.is_empty() {
        println!("(no aliases match the requested filters)");
        return Ok(());
    }

    if repo_only {
        // De-dup repos in case a single repo is referenced by
        // multiple AliasEntry rows in some future expansion.
        let mut seen = std::collections::HashSet::new();
        for e in &entries {
            if seen.insert(e.repo) {
                println!("{}", e.repo);
            }
        }
        return Ok(());
    }

    let mut current_family = "";
    for e in &entries {
        if e.family != current_family {
            if !current_family.is_empty() {
                println!();
            }
            println!("── {} ──", e.family);
            current_family = e.family;
        }
        let aliases = e.aliases.join(", ");
        let gate = if e.gated { " [gated]" } else { "" };
        println!("  {aliases}");
        println!("    → {} ({}){}", e.repo, e.kind, gate);
        println!("    {}", e.note);
    }
    println!();
    println!(
        "Total: {} alias group{} ({} accepted in --model).",
        entries.len(),
        if entries.len() == 1 { "" } else { "s" },
        entries.iter().map(|e| e.aliases.len()).sum::<usize>(),
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn print_aliases_runs_without_filter() {
        // Smoke test: exercising the formatter against the live
        // ALIAS_TABLE catches a panic if the heading-grouping
        // logic ever desyncs from the data shape (e.g. a future
        // entry with an empty family string).
        print_aliases(None, false, false).unwrap();
    }

    #[test]
    fn print_aliases_runs_with_each_filter() {
        print_aliases(Some("flux"), false, false).unwrap();
        print_aliases(None, true, false).unwrap();
        print_aliases(None, false, true).unwrap();
        // Combined: gated + repo-only against a single family.
        print_aliases(Some("Flux"), true, true).unwrap();
    }

    #[test]
    fn print_aliases_runs_with_empty_filter_result() {
        // Substring that matches no family — exercises the
        // "(no aliases match…)" branch.
        print_aliases(Some("no-such-family-xyz"), false, false).unwrap();
    }
}

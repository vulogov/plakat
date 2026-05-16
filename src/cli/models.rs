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
    }
}

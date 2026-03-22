use anyhow::Result;
use axon_pack_rs::{build_from_plan, load_bundle, validate_bundle};
use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "axon-pack-rs", about = "AXON writer and validator")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    Build {
        #[arg(long)]
        plan: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    Validate {
        #[arg(long)]
        bundle: PathBuf,
    },
    Inspect {
        #[arg(long)]
        bundle: PathBuf,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Build { plan, output } => {
            let summary = build_from_plan(&plan, &output)?;
            println!("{}", serde_json::to_string_pretty(&summary)?);
        }
        Command::Validate { bundle } => {
            let summary = validate_bundle(&bundle)?;
            println!("{}", serde_json::to_string_pretty(&summary)?);
        }
        Command::Inspect { bundle } => {
            let summary = load_bundle(&bundle)?;
            println!("{}", serde_json::to_string_pretty(&summary)?);
        }
    }
    Ok(())
}


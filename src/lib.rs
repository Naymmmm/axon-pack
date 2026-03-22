use anyhow::{Context, Result, anyhow, bail};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;
use xxhash_rust::xxh64::Xxh64;

pub const MAGIC: [u8; 4] = *b"AXON";
pub const VERSION_MAJOR: u16 = 2;
pub const VERSION_MINOR: u16 = 0;

pub const FLAG_HAS_CODEBOOKS: u32 = 1 << 0;
pub const FLAG_HAS_OUTLIER_SPINE: u32 = 1 << 1;
pub const FLAG_STREAM_ORDERED: u32 = 1 << 2;
pub const FLAG_HAS_HW_HINTS: u32 = 1 << 3;
pub const FLAG_HEADER_ZSTD: u32 = 1 << 4;
pub const FLAG_SCALES_INTERLEAVED: u32 = 1 << 5;
pub const FLAG_HAS_CHECKSUMS: u32 = 1 << 6;
pub const FLAG_MXQ_V2: u32 = 1 << 7;
pub const FLAG_HAS_BOOT_REGION: u32 = 1 << 8;
pub const FLAG_HAS_KV_HINTS: u32 = 1 << 9;
pub const FLAG_HAS_DEP_GRAPH: u32 = 1 << 10;
pub const FLAG_HAS_SPECULATIVE_DRAFT: u32 = 1 << 11;
pub const FLAG_HAS_EXPERT_DEDUP: u32 = 1 << 12;
pub const FLAG_HAS_LORA_DELTA: u32 = 1 << 13;
pub const FLAG_PER_HEAD_QUANT: u32 = 1 << 14;
pub const FLAG_NF_QUANT: u32 = 1 << 15;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BundleManifest {
    pub manifest_version: String,
    pub task: String,
    pub model_file: String,
    #[serde(default)]
    pub config_file: Option<String>,
    #[serde(default)]
    pub generation_config_file: Option<String>,
    #[serde(default)]
    pub tokenizer: Option<TokenizerManifest>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizerManifest {
    pub kind: String,
    pub files: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub hidden_dim: u64,
    pub intermediate_dim: u64,
    pub num_layers: u64,
    pub num_attention_heads: u64,
    pub num_kv_heads: u64,
    pub head_dim: u64,
    pub context_length: u64,
    pub vocab_size: u64,
    pub rope: RopeConfig,
    #[serde(default)]
    pub total_parameter_count: Option<u64>,
    #[serde(default)]
    pub active_parameter_count: Option<u64>,
    #[serde(default)]
    pub moe: Option<MoeConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoeConfig {
    pub num_experts: u64,
    pub experts_per_token: u64,
    pub expert_intermediate_dim: u64,
    #[serde(default)]
    pub num_shared_experts: Option<u64>,
    #[serde(default)]
    pub shared_expert_intermediate_dim: Option<u64>,
    #[serde(default)]
    pub router_aux_loss_coef: Option<f64>,
    #[serde(default)]
    pub expert_layer_frequency: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RopeConfig {
    #[serde(rename = "type")]
    pub rope_type: String,
    #[serde(default)]
    pub theta: Option<f64>,
    #[serde(default)]
    pub scaling: Option<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeConfig {
    #[serde(default)]
    pub bos_token_id: Option<i64>,
    #[serde(default)]
    pub eos_token_id: Option<i64>,
    #[serde(default)]
    pub pad_token_id: Option<i64>,
    pub default_generation: GenerationConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationConfig {
    pub temperature: f64,
    pub top_p: f64,
    pub top_k: u64,
    pub max_new_tokens: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceInfo {
    pub format: String,
    pub identifier: String,
    pub conversion_tool: String,
    pub conversion_time_utc: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationInfo {
    pub dataset: String,
    pub tokens: u64,
    pub perplexity_fp16: f64,
    pub perplexity_axon: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompactMoeInfo {
    pub num_experts: u64,
    pub active_experts: u64,
    pub expert_hidden_dim: u64,
    #[serde(default)]
    pub expert_similarity_dedup: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KvCacheLayerHint {
    pub layer: i64,
    pub dtype: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KvCacheHints {
    pub default_dtype: String,
    #[serde(default)]
    pub per_layer: Vec<KvCacheLayerHint>,
    pub max_seq_kv_budget_mb: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorDependencyGraph {
    #[serde(default)]
    pub parallel_groups: Vec<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeculativeDraftInfo {
    pub arch: String,
    pub hidden_dim: u64,
    pub num_layers: u64,
    pub vocab_size: u64,
    pub draft_bytes: u64,
    pub draft_offset: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmoothQuantScaleDescriptor {
    pub offset: u64,
    pub channels: u64,
    pub dtype: String,
    pub size: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertDedupInfo {
    pub region_offset: u64,
    pub region_bytes: u64,
    #[serde(default)]
    pub canonical_map: BTreeMap<String, String>,
    pub corrections_offset: u64,
    pub similarity_threshold: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoraInfo {
    pub base_model: String,
    pub base_hash: String,
    pub rank: u64,
    pub alpha: u64,
    #[serde(default)]
    pub target_modules: Vec<String>,
    pub region_offset: u64,
    pub region_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NumaHints {
    pub num_nodes: u64,
    #[serde(default)]
    pub tensor_node_map: BTreeMap<String, u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorDescriptor {
    pub shape: Vec<u64>,
    pub dtype: String,
    #[serde(default)]
    pub bits: Option<u8>,
    #[serde(default)]
    pub group_size: Option<u64>,
    #[serde(default)]
    pub source_tensor_name: Option<String>,
    #[serde(default)]
    pub data_offset: u64,
    #[serde(default)]
    pub data_bytes: u64,
    #[serde(default)]
    pub scale_interleaved: bool,
    #[serde(default)]
    pub outlier_indices_offset: Option<u64>,
    #[serde(default)]
    pub outlier_count: Option<u64>,
    #[serde(default)]
    pub sensitivity_score: Option<f64>,
    pub stream_order: u32,
    #[serde(default)]
    pub per_head_bits: Option<Vec<u8>>,
    #[serde(default)]
    pub nf_scale_fp16: bool,
    #[serde(default)]
    pub smoothquant_scale: Option<String>,
    #[serde(default)]
    pub prefetch_priority: Option<f64>,
    #[serde(default)]
    pub codebook_id: Option<String>,
    #[serde(default)]
    pub vq_dim: Option<u64>,
    #[serde(default)]
    pub dedup_canonical: Option<String>,
    #[serde(default)]
    pub dedup_correction_offset: Option<u64>,
    #[serde(default)]
    pub dedup_correction_count: Option<u64>,
    #[serde(default)]
    pub lora_rank: Option<u64>,
    #[serde(default)]
    pub lora_alpha: Option<u64>,
    #[serde(default)]
    pub target: Option<String>,
    #[serde(default)]
    pub checksum_xxh64: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodebookDescriptor {
    pub offset: u64,
    pub entries: u64,
    pub dim: u64,
    pub dtype: String,
    pub size: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareHint {
    pub kernel: String,
    #[serde(default)]
    pub tile: Option<Vec<u64>>,
    #[serde(default)]
    pub unroll: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Metadata {
    pub format: String,
    pub version: String,
    pub task: String,
    pub architecture: String,
    pub model_family: String,
    pub model: ModelConfig,
    pub runtime: RuntimeConfig,
    pub source: SourceInfo,
    #[serde(default)]
    pub total_params: Option<u64>,
    #[serde(default)]
    pub active_params: Option<u64>,
    #[serde(default)]
    pub moe: Option<CompactMoeInfo>,
    #[serde(default)]
    pub boot_region_bytes: Option<u64>,
    #[serde(default)]
    pub kv_cache_hints: Option<KvCacheHints>,
    #[serde(default)]
    pub tensor_dep_graph: Option<TensorDependencyGraph>,
    #[serde(default)]
    pub speculative_draft: Option<SpeculativeDraftInfo>,
    #[serde(default)]
    pub smoothquant_scales: BTreeMap<String, SmoothQuantScaleDescriptor>,
    #[serde(default)]
    pub expert_dedup: Option<ExpertDedupInfo>,
    #[serde(default)]
    pub lora: Option<LoraInfo>,
    #[serde(default)]
    pub numa_hints: Option<NumaHints>,
    pub avg_bits_per_weight: f64,
    pub quant_method: String,
    #[serde(default)]
    pub calibration: Option<CalibrationInfo>,
    pub tensors: BTreeMap<String, TensorDescriptor>,
    #[serde(default)]
    pub codebooks: BTreeMap<String, CodebookDescriptor>,
    #[serde(default)]
    pub hw_hints: BTreeMap<String, HardwareHint>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ByteSource {
    pub path: String,
    pub offset: u64,
    pub length: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssetCopy {
    pub source: String,
    pub dest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildPlan {
    pub manifest: BundleManifest,
    pub metadata: Metadata,
    #[serde(default)]
    pub boot_cutoff: Option<u32>,
    pub tensor_sources: BTreeMap<String, ByteSource>,
    #[serde(default)]
    pub outlier_sources: BTreeMap<String, ByteSource>,
    #[serde(default)]
    pub codebook_sources: BTreeMap<String, ByteSource>,
    #[serde(default)]
    pub expert_dedup_sources: BTreeMap<String, ByteSource>,
    #[serde(default)]
    pub speculative_draft_source: Option<ByteSource>,
    #[serde(default)]
    pub lora_sources: BTreeMap<String, ByteSource>,
    #[serde(default)]
    pub assets: Vec<AssetCopy>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BundleHeader {
    pub version_major: u16,
    pub version_minor: u16,
    pub flags: u32,
    pub header_len: u32,
    pub data_offset: u64,
    pub outlier_offset: u64,
    pub codebook_offset: u64,
    pub tail_offset: u64,
    pub boot_cutoff: u8,
    pub speculative_offset: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BundleSummary {
    pub manifest: BundleManifest,
    pub metadata: Metadata,
    pub file_size: u64,
    pub flags: u32,
}

pub fn build_from_plan(plan_path: &Path, output_dir: &Path) -> Result<BundleSummary> {
    let plan_text = fs::read_to_string(plan_path)
        .with_context(|| format!("failed to read build plan {}", plan_path.display()))?;
    let mut plan: BuildPlan = serde_json::from_str(&plan_text).context("failed to parse build plan JSON")?;
    normalize_metadata(&mut plan.metadata)?;
    validate_plan_sources(&plan)?;

    assign_tensor_layouts(&mut plan)?;
    assign_outlier_layouts(&mut plan)?;
    assign_codebook_layouts(&mut plan)?;
    assign_expert_dedup_layouts(&mut plan)?;
    assign_lora_layouts(&mut plan)?;
    plan.metadata.avg_bits_per_weight = compute_avg_bits_per_weight(&plan);

    let data_region_size = compute_tensor_region_size(&plan.metadata, &plan.tensor_sources)?;
    let outlier_region_size = compute_named_region_size(
        plan.metadata
            .tensors
            .iter()
            .filter_map(|(name, tensor)| tensor.outlier_indices_offset.map(|offset| (name.clone(), offset)))
            .collect(),
        &plan.outlier_sources,
    )?;
    let codebook_region_size = compute_named_region_size(
        plan.metadata
            .codebooks
            .iter()
            .map(|(name, descriptor)| (name.clone(), descriptor.offset))
            .collect(),
        &plan.codebook_sources,
    )?;
    let expert_dedup_region_size = compute_dedup_region_size(&plan.metadata, &plan.expert_dedup_sources)?;
    let lora_region_size = compute_lora_region_size(&plan.metadata, &plan.lora_sources)?;
    let speculative_draft_len = plan.speculative_draft_source.as_ref().map(|source| source.length).unwrap_or(0);

    let mut final_header = None;
    let mut final_metadata_bytes = None;
    for _ in 0..8 {
        let metadata_bytes = serde_json::to_vec(&plan.metadata).context("failed to serialize AXON metadata")?;
        let data_offset = align64(64 + metadata_bytes.len() as u64);
        let outlier_offset = if outlier_region_size > 0 {
            align64(data_offset + data_region_size)
        } else {
            0
        };
        let codebook_offset = if codebook_region_size > 0 {
            let base = if outlier_offset > 0 {
                outlier_offset + outlier_region_size
            } else {
                data_offset + data_region_size
            };
            align64(base)
        } else {
            0
        };
        let expert_dedup_offset = if expert_dedup_region_size > 0 {
            let base = if codebook_offset > 0 {
                codebook_offset + codebook_region_size
            } else if outlier_offset > 0 {
                outlier_offset + outlier_region_size
            } else {
                data_offset + data_region_size
            };
            align64(base)
        } else {
            0
        };
        let speculative_offset = if speculative_draft_len > 0 {
            let base = if expert_dedup_offset > 0 {
                expert_dedup_offset + expert_dedup_region_size
            } else if codebook_offset > 0 {
                codebook_offset + codebook_region_size
            } else if outlier_offset > 0 {
                outlier_offset + outlier_region_size
            } else {
                data_offset + data_region_size
            };
            align64(base)
        } else {
            0
        };
        let lora_offset = if lora_region_size > 0 {
            let base = if speculative_offset > 0 {
                speculative_offset + speculative_draft_len
            } else if expert_dedup_offset > 0 {
                expert_dedup_offset + expert_dedup_region_size
            } else if codebook_offset > 0 {
                codebook_offset + codebook_region_size
            } else if outlier_offset > 0 {
                outlier_offset + outlier_region_size
            } else {
                data_offset + data_region_size
            };
            align64(base)
        } else {
            0
        };

        if let Some(dedup) = plan.metadata.expert_dedup.as_mut() {
            dedup.region_offset = expert_dedup_offset;
            dedup.region_bytes = expert_dedup_region_size;
            dedup.corrections_offset = 0;
        }
        if let Some(speculative) = plan.metadata.speculative_draft.as_mut() {
            speculative.draft_bytes = speculative_draft_len;
            speculative.draft_offset = 0;
        }
        if let Some(lora) = plan.metadata.lora.as_mut() {
            lora.region_offset = lora_offset;
            lora.region_bytes = lora_region_size;
        }

        let boot_bytes = if let Some(cutoff) = plan.boot_cutoff {
            compute_boot_region_bytes(&plan.metadata, cutoff)
        } else {
            None
        };
        plan.metadata.boot_region_bytes = boot_bytes;

        let stabilized = serde_json::to_vec(&plan.metadata).context("failed to serialize AXON metadata")?;
        let stabilized_data_offset = align64(64 + stabilized.len() as u64);
        let boot_cutoff = boot_bytes.map(|_| plan.boot_cutoff.unwrap_or(0).min(u8::MAX as u32) as u8).unwrap_or(0);
        let tail_offset = if let Some(bytes) = boot_bytes {
            if bytes > 0 { data_offset + bytes } else { 0 }
        } else {
            0
        };
        let header = build_header(
            derive_flags(&plan.metadata, &plan),
            stabilized.len() as u32,
            stabilized_data_offset,
            outlier_offset,
            codebook_offset,
            tail_offset,
            boot_cutoff,
            speculative_offset,
        );
        if stabilized_data_offset == data_offset {
            final_header = Some(header);
            final_metadata_bytes = Some(stabilized);
            break;
        }
    }

    let header = final_header.ok_or_else(|| anyhow!("failed to stabilize AXON metadata layout"))?;
    let metadata_bytes = final_metadata_bytes.ok_or_else(|| anyhow!("missing serialized AXON metadata"))?;
    let file_size = write_bundle(&plan, output_dir, header, &metadata_bytes)?;
    write_manifest_and_assets(&plan, output_dir)?;

    let bundle = load_bundle(output_dir)?;
    if bundle.file_size != file_size {
        bail!("bundle size mismatch after write");
    }
    Ok(bundle)
}

pub fn load_bundle(bundle_dir: &Path) -> Result<BundleSummary> {
    let manifest_path = bundle_dir.join("manifest.json");
    let manifest_text = fs::read_to_string(&manifest_path)
        .with_context(|| format!("failed to read {}", manifest_path.display()))?;
    let manifest: BundleManifest = serde_json::from_str(&manifest_text).context("failed to parse manifest.json")?;
    validate_manifest_files(bundle_dir, &manifest)?;

    let model_path = bundle_dir.join(&manifest.model_file);
    let mut file = File::open(&model_path).with_context(|| format!("failed to open {}", model_path.display()))?;
    let file_size = file.metadata()?.len();
    let header = read_header(&mut file)?;
    validate_header(&header)?;
    let metadata_bytes = read_metadata_bytes(&mut file, &header)?;
    let metadata: Metadata = serde_json::from_slice(&metadata_bytes).context("failed to parse metadata JSON")?;
    validate_metadata(&metadata, &header)?;
    validate_tensor_regions(&mut file, &header, &metadata, file_size)?;
    validate_outlier_regions(&header, &metadata, file_size)?;
    validate_codebook_regions(&header, &metadata, file_size)?;
    validate_expert_dedup_regions(&metadata, file_size)?;
    validate_speculative_region(&header, &metadata, file_size)?;

    Ok(BundleSummary {
        manifest,
        metadata,
        file_size,
        flags: header.flags,
    })
}

pub fn validate_bundle(bundle_dir: &Path) -> Result<BundleSummary> {
    load_bundle(bundle_dir)
}

fn normalize_metadata(metadata: &mut Metadata) -> Result<()> {
    if metadata.format.is_empty() {
        metadata.format = "axon".to_string();
    }
    if metadata.version.is_empty() {
        metadata.version = "2.0.0-draft".to_string();
    }
    if metadata.task.is_empty() {
        metadata.task = "causal_lm".to_string();
    }
    if metadata.quant_method.is_empty() {
        metadata.quant_method = "none".to_string();
    }
    if metadata.total_params.is_none() {
        metadata.total_params = metadata.model.total_parameter_count;
    }
    if metadata.active_params.is_none() {
        metadata.active_params = metadata.model.active_parameter_count;
    }
    if metadata.moe.is_none() {
        metadata.moe = metadata.model.moe.as_ref().map(|moe| CompactMoeInfo {
            num_experts: moe.num_experts,
            active_experts: moe.experts_per_token,
            expert_hidden_dim: moe.expert_intermediate_dim,
            expert_similarity_dedup: metadata.expert_dedup.is_some(),
        });
    }
    validate_metadata_shape(metadata)
}

fn validate_metadata_shape(metadata: &Metadata) -> Result<()> {
    if metadata.format != "axon" {
        bail!("metadata format must be axon");
    }
    if metadata.task != "causal_lm" {
        bail!("unsupported task type {}", metadata.task);
    }
    if metadata.tensors.is_empty() {
        bail!("metadata contains no tensors");
    }
    validate_model_config(&metadata.model)?;
    if let Some(total) = metadata.total_params {
        if total == 0 {
            bail!("metadata total_params must be greater than zero");
        }
        if let Some(model_total) = metadata.model.total_parameter_count {
            if total != model_total {
                bail!("metadata total_params does not match model.total_parameter_count");
            }
        }
    }
    if let Some(active) = metadata.active_params {
        if active == 0 {
            bail!("metadata active_params must be greater than zero");
        }
        if let Some(total) = metadata.total_params {
            if active > total {
                bail!("metadata active_params exceeds total_params");
            }
        }
        if let Some(model_active) = metadata.model.active_parameter_count {
            if active != model_active {
                bail!("metadata active_params does not match model.active_parameter_count");
            }
        }
    }
    if let Some(moe) = &metadata.moe {
        if moe.num_experts == 0 || moe.active_experts == 0 || moe.expert_hidden_dim == 0 {
            bail!("metadata moe fields must be greater than zero");
        }
        if moe.active_experts > moe.num_experts {
            bail!("metadata moe.active_experts exceeds moe.num_experts");
        }
    }
    validate_v2_metadata(metadata)?;
    Ok(())
}

fn validate_v2_metadata(metadata: &Metadata) -> Result<()> {
    if let Some(kv_hints) = &metadata.kv_cache_hints {
        if kv_hints.default_dtype.is_empty() {
            bail!("kv_cache_hints.default_dtype must not be empty");
        }
    }
    if let Some(graph) = &metadata.tensor_dep_graph {
        for group in &graph.parallel_groups {
            if group.is_empty() {
                bail!("tensor_dep_graph.parallel_groups must not contain empty groups");
            }
        }
    }
    if let Some(speculative) = &metadata.speculative_draft {
        if speculative.draft_bytes == 0 {
            bail!("speculative_draft.draft_bytes must be greater than zero");
        }
    }
    for (scale_name, descriptor) in &metadata.smoothquant_scales {
        if descriptor.channels == 0 || descriptor.size == 0 {
            bail!("smoothquant scale {scale_name} has invalid size");
        }
    }
    if let Some(dedup) = &metadata.expert_dedup {
        if dedup.region_bytes == 0 {
            bail!("expert_dedup.region_bytes must be greater than zero");
        }
    }
    if let Some(lora) = &metadata.lora {
        if lora.rank == 0 || lora.alpha == 0 || lora.region_bytes == 0 {
            bail!("lora metadata contains invalid zero values");
        }
    }
    if let Some(numa) = &metadata.numa_hints {
        if numa.num_nodes == 0 {
            bail!("numa_hints.num_nodes must be greater than zero");
        }
    }
    Ok(())
}

fn validate_model_config(model: &ModelConfig) -> Result<()> {
    if let Some(total) = model.total_parameter_count {
        if total == 0 {
            bail!("model total_parameter_count must be greater than zero");
        }
    }
    if let Some(active) = model.active_parameter_count {
        if active == 0 {
            bail!("model active_parameter_count must be greater than zero");
        }
        if let Some(total) = model.total_parameter_count {
            if active > total {
                bail!("model active_parameter_count exceeds total_parameter_count");
            }
        }
    }
    if let Some(moe) = &model.moe {
        if moe.num_experts == 0 {
            bail!("model moe.num_experts must be greater than zero");
        }
        if moe.experts_per_token == 0 {
            bail!("model moe.experts_per_token must be greater than zero");
        }
        if moe.experts_per_token > moe.num_experts {
            bail!("model moe.experts_per_token exceeds moe.num_experts");
        }
        if moe.expert_intermediate_dim == 0 {
            bail!("model moe.expert_intermediate_dim must be greater than zero");
        }
        if let Some(shared) = moe.num_shared_experts {
            if shared == 0 {
                bail!("model moe.num_shared_experts must be greater than zero when present");
            }
        }
        if let Some(shared_dim) = moe.shared_expert_intermediate_dim {
            if shared_dim == 0 {
                bail!("model moe.shared_expert_intermediate_dim must be greater than zero when present");
            }
        }
        if let Some(layer_frequency) = moe.expert_layer_frequency {
            if layer_frequency == 0 {
                bail!("model moe.expert_layer_frequency must be greater than zero when present");
            }
        }
    }
    Ok(())
}

fn validate_plan_sources(plan: &BuildPlan) -> Result<()> {
    for (name, tensor) in &plan.metadata.tensors {
        if tensor.dedup_canonical.is_some() {
            if tensor.dedup_correction_offset.is_some() || tensor.dedup_correction_count.is_some() {
                let source = plan
                    .expert_dedup_sources
                    .get(name)
                    .ok_or_else(|| anyhow!("missing expert dedup correction source for {name}"))?;
                ensure_source(source)?;
            }
            continue;
        }
        let source = if plan.metadata.lora.is_some() && tensor.target.is_some() {
            plan.lora_sources
                .get(name)
                .ok_or_else(|| anyhow!("missing LoRA source for {name}"))?
        } else {
            plan.tensor_sources
                .get(name)
                .ok_or_else(|| anyhow!("missing tensor source for {name}"))?
        };
        ensure_source(source)?;
        match tensor.dtype.as_str() {
            "fp16" | "bf16" | "fp32" => {}
            "axon_mxq" | "axon_nf2" | "axon_nf3" => {
                if tensor.bits.is_none() || tensor.group_size.is_none() {
                    bail!("quantized tensor {name} is missing bits or group_size");
                }
                if matches!(tensor.dtype.as_str(), "axon_nf2" | "axon_nf3") && !tensor.nf_scale_fp16 {
                    bail!("NF tensor {name} must set nf_scale_fp16");
                }
            }
            "axon_vq" => {
                let codebook_id = tensor
                    .codebook_id
                    .as_deref()
                    .ok_or_else(|| anyhow!("VQ tensor {name} is missing codebook_id"))?;
                if tensor.vq_dim.is_none() {
                    bail!("VQ tensor {name} is missing vq_dim");
                }
                if !plan.metadata.codebooks.contains_key(codebook_id) {
                    bail!("VQ tensor {name} references missing codebook {codebook_id}");
                }
            }
            other => bail!("unsupported tensor dtype {other} for {name}"),
        }
        if let Some(scale_name) = &tensor.smoothquant_scale {
            if !plan.metadata.smoothquant_scales.contains_key(scale_name) {
                bail!("tensor {name} references missing smoothquant scale {scale_name}");
            }
        }
        if let Some(priority) = tensor.prefetch_priority {
            if !(0.0..=1.0).contains(&priority) {
                bail!("tensor {name} has invalid prefetch_priority");
            }
        }
        if tensor.outlier_indices_offset.is_some() || tensor.outlier_count.is_some() {
            let source = plan
                .outlier_sources
                .get(name)
                .ok_or_else(|| anyhow!("missing outlier source for {name}"))?;
            ensure_source(source)?;
        }
    }
    for (codebook_id, _) in &plan.metadata.codebooks {
        let source = plan
            .codebook_sources
            .get(codebook_id)
            .ok_or_else(|| anyhow!("missing codebook source for {codebook_id}"))?;
        ensure_source(source)?;
    }
    if let Some(source) = &plan.speculative_draft_source {
        ensure_source(source)?;
    }
    Ok(())
}

fn ensure_source(source: &ByteSource) -> Result<()> {
    let path = Path::new(&source.path);
    let metadata = fs::metadata(path).with_context(|| format!("failed to stat {}", path.display()))?;
    let end = source
        .offset
        .checked_add(source.length)
        .ok_or_else(|| anyhow!("source length overflow for {}", path.display()))?;
    if end > metadata.len() {
        bail!("source slice exceeds file bounds for {}", path.display());
    }
    Ok(())
}

fn assign_tensor_layouts(plan: &mut BuildPlan) -> Result<()> {
    let ordered = ordered_tensor_names(&plan.metadata);
    let mut relative_offset = 0_u64;
    for tensor_name in ordered {
        let descriptor = plan
            .metadata
            .tensors
            .get_mut(&tensor_name)
            .ok_or_else(|| anyhow!("missing tensor descriptor for {tensor_name}"))?;
        if descriptor.dedup_canonical.is_some() || (plan.metadata.lora.is_some() && descriptor.target.is_some()) {
            continue;
        }
        let source = plan
            .tensor_sources
            .get(&tensor_name)
            .ok_or_else(|| anyhow!("missing tensor source for {tensor_name}"))?;
        relative_offset = align64(relative_offset);
        descriptor.data_offset = relative_offset;
        descriptor.data_bytes = source.length;
        descriptor.checksum_xxh64 = Some(compute_source_checksum(source)?);
        relative_offset += source.length;
    }
    Ok(())
}

fn assign_outlier_layouts(plan: &mut BuildPlan) -> Result<()> {
    let ordered = ordered_tensor_names(&plan.metadata);
    let mut relative_offset = 0_u64;
    for tensor_name in ordered {
        if !plan.outlier_sources.contains_key(&tensor_name) {
            continue;
        }
        let descriptor = plan
            .metadata
            .tensors
            .get_mut(&tensor_name)
            .ok_or_else(|| anyhow!("missing tensor descriptor for {tensor_name}"))?;
        relative_offset = align64(relative_offset);
        descriptor.outlier_indices_offset = Some(relative_offset);
        relative_offset += plan.outlier_sources[&tensor_name].length;
    }
    Ok(())
}

fn assign_expert_dedup_layouts(plan: &mut BuildPlan) -> Result<()> {
    let ordered = ordered_tensor_names(&plan.metadata);
    let mut relative_offset = 0_u64;
    for tensor_name in ordered {
        let descriptor = plan
            .metadata
            .tensors
            .get_mut(&tensor_name)
            .ok_or_else(|| anyhow!("missing tensor descriptor for {tensor_name}"))?;
        if descriptor.dedup_canonical.is_none() {
            continue;
        }
        let Some(source) = plan.expert_dedup_sources.get(&tensor_name) else {
            continue;
        };
        relative_offset = align64(relative_offset);
        descriptor.dedup_correction_offset = Some(relative_offset);
        relative_offset += source.length;
    }
    if let Some(dedup) = plan.metadata.expert_dedup.as_mut() {
        dedup.corrections_offset = 0;
        dedup.region_bytes = relative_offset;
    }
    Ok(())
}

fn assign_codebook_layouts(plan: &mut BuildPlan) -> Result<()> {
    let mut relative_offset = 0_u64;
    let codebook_ids: Vec<_> = plan.metadata.codebooks.keys().cloned().collect();
    for codebook_id in codebook_ids {
        let descriptor = plan
            .metadata
            .codebooks
            .get_mut(&codebook_id)
            .ok_or_else(|| anyhow!("missing codebook descriptor for {codebook_id}"))?;
        let source = plan
            .codebook_sources
            .get(&codebook_id)
            .ok_or_else(|| anyhow!("missing codebook source for {codebook_id}"))?;
        relative_offset = align64(relative_offset);
        descriptor.offset = relative_offset;
        descriptor.size = source.length;
        relative_offset += source.length;
    }
    Ok(())
}

fn assign_lora_layouts(plan: &mut BuildPlan) -> Result<()> {
    if plan.metadata.lora.is_none() {
        return Ok(());
    }
    let ordered = ordered_tensor_names(&plan.metadata);
    let mut relative_offset = 0_u64;
    for tensor_name in ordered {
        let descriptor = plan
            .metadata
            .tensors
            .get_mut(&tensor_name)
            .ok_or_else(|| anyhow!("missing tensor descriptor for {tensor_name}"))?;
        if descriptor.target.is_none() {
            continue;
        }
        let source = plan
            .lora_sources
            .get(&tensor_name)
            .ok_or_else(|| anyhow!("missing LoRA source for {tensor_name}"))?;
        relative_offset = align64(relative_offset);
        descriptor.data_offset = relative_offset;
        descriptor.data_bytes = source.length;
        descriptor.checksum_xxh64 = Some(compute_source_checksum(source)?);
        relative_offset += source.length;
    }
    if let Some(lora) = plan.metadata.lora.as_mut() {
        lora.region_bytes = relative_offset;
    }
    Ok(())
}

fn ordered_tensor_names(metadata: &Metadata) -> Vec<String> {
    let mut entries: Vec<_> = metadata
        .tensors
        .iter()
        .map(|(name, tensor)| (name.clone(), tensor.stream_order))
        .collect();
    entries.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
    entries.into_iter().map(|(name, _)| name).collect()
}

fn is_lora_tensor(metadata: &Metadata, tensor: &TensorDescriptor) -> bool {
    metadata.lora.is_some() && tensor.target.is_some()
}

fn compute_source_checksum(source: &ByteSource) -> Result<String> {
    let mut file = File::open(&source.path).with_context(|| format!("failed to open {}", source.path))?;
    file.seek(SeekFrom::Start(source.offset))?;
    let mut remaining = source.length;
    let mut buffer = vec![0_u8; 64 * 1024];
    let mut hasher = Xxh64::new(0);
    while remaining > 0 {
        let read_len = remaining.min(buffer.len() as u64) as usize;
        file.read_exact(&mut buffer[..read_len])?;
        hasher.update(&buffer[..read_len]);
        remaining -= read_len as u64;
    }
    Ok(format!("{:016x}", hasher.digest()))
}

fn compute_avg_bits_per_weight(plan: &BuildPlan) -> f64 {
    let mut total_bits = 0_f64;
    let mut total_params = 0_f64;
    for (name, tensor) in &plan.metadata.tensors {
        let params = tensor.shape.iter().product::<u64>() as f64;
        if params == 0.0 {
            continue;
        }
        total_params += params;
        total_bits += tensor.data_bytes as f64 * 8.0;
        if let Some(source) = plan.outlier_sources.get(name) {
            total_bits += source.length as f64 * 8.0;
        }
        if let Some(source) = plan.expert_dedup_sources.get(name) {
            total_bits += source.length as f64 * 8.0;
        }
        if let Some(codebook_id) = &tensor.codebook_id {
            if let Some(source) = plan.codebook_sources.get(codebook_id) {
                total_bits += source.length as f64 * 8.0;
            }
        }
        if let Some(source) = plan.lora_sources.get(name) {
            total_bits += source.length as f64 * 8.0;
        }
    }
    if total_params == 0.0 {
        0.0
    } else {
        total_bits / total_params
    }
}

fn derive_flags(metadata: &Metadata, plan: &BuildPlan) -> u32 {
    let mut flags = FLAG_HAS_CHECKSUMS | FLAG_STREAM_ORDERED;
    if !metadata.codebooks.is_empty() && !plan.codebook_sources.is_empty() {
        flags |= FLAG_HAS_CODEBOOKS;
    }
    if !plan.outlier_sources.is_empty() {
        flags |= FLAG_HAS_OUTLIER_SPINE;
    }
    if !metadata.hw_hints.is_empty() {
        flags |= FLAG_HAS_HW_HINTS;
    }
    if metadata
        .tensors
        .values()
        .all(|tensor| !matches!(tensor.dtype.as_str(), "axon_mxq" | "axon_nf2" | "axon_nf3") || tensor.scale_interleaved)
    {
        flags |= FLAG_SCALES_INTERLEAVED;
    }
    if metadata.boot_region_bytes.unwrap_or(0) > 0 {
        flags |= FLAG_HAS_BOOT_REGION;
    }
    if metadata.kv_cache_hints.is_some() {
        flags |= FLAG_HAS_KV_HINTS;
    }
    if metadata.tensor_dep_graph.is_some() {
        flags |= FLAG_HAS_DEP_GRAPH;
    }
    if metadata.speculative_draft.is_some() {
        flags |= FLAG_HAS_SPECULATIVE_DRAFT;
    }
    if metadata.expert_dedup.is_some() {
        flags |= FLAG_HAS_EXPERT_DEDUP;
    }
    if metadata.lora.is_some() {
        flags |= FLAG_HAS_LORA_DELTA;
    }
    if metadata.tensors.values().any(|tensor| tensor.per_head_bits.is_some()) {
        flags |= FLAG_PER_HEAD_QUANT;
    }
    if metadata
        .tensors
        .values()
        .any(|tensor| matches!(tensor.dtype.as_str(), "axon_nf2" | "axon_nf3"))
    {
        flags |= FLAG_NF_QUANT;
    }
    flags
}

fn compute_tensor_region_size(metadata: &Metadata, sources: &BTreeMap<String, ByteSource>) -> Result<u64> {
    let mut end = 0_u64;
    for (name, tensor) in &metadata.tensors {
        if tensor.dedup_canonical.is_some() || is_lora_tensor(metadata, tensor) {
            continue;
        }
        let _ = sources
            .get(name)
            .ok_or_else(|| anyhow!("missing tensor source for {name}"))?;
        end = end.max(tensor.data_offset + tensor.data_bytes);
    }
    Ok(end)
}

fn compute_dedup_region_size(metadata: &Metadata, sources: &BTreeMap<String, ByteSource>) -> Result<u64> {
    let mut end = 0_u64;
    for (name, tensor) in &metadata.tensors {
        let Some(offset) = tensor.dedup_correction_offset else {
            continue;
        };
        let source = sources
            .get(name)
            .ok_or_else(|| anyhow!("missing expert dedup source for {name}"))?;
        end = end.max(offset + source.length);
    }
    Ok(end)
}

fn compute_lora_region_size(metadata: &Metadata, sources: &BTreeMap<String, ByteSource>) -> Result<u64> {
    let mut end = 0_u64;
    for (name, tensor) in &metadata.tensors {
        if !is_lora_tensor(metadata, tensor) {
            continue;
        }
        let source = sources
            .get(name)
            .ok_or_else(|| anyhow!("missing LoRA source for {name}"))?;
        end = end.max(tensor.data_offset + source.length);
    }
    Ok(end)
}

fn compute_boot_region_bytes(metadata: &Metadata, cutoff: u32) -> Option<u64> {
    let mut tail_start = None;
    let mut data_end = 0_u64;
    for tensor in metadata.tensors.values() {
        if tensor.dedup_canonical.is_some() || is_lora_tensor(metadata, tensor) {
            continue;
        }
        data_end = data_end.max(tensor.data_offset + tensor.data_bytes);
        if tail_start.is_none() && tensor.stream_order >= cutoff {
            tail_start = Some(tensor.data_offset);
        }
    }
    let bytes = tail_start.unwrap_or(data_end);
    if bytes == 0 { None } else { Some(bytes) }
}

fn compute_named_region_size(
    layout: BTreeMap<String, u64>,
    sources: &BTreeMap<String, ByteSource>,
) -> Result<u64> {
    let mut end = 0_u64;
    for (name, offset) in layout {
        let source = sources
            .get(&name)
            .ok_or_else(|| anyhow!("missing region source for {name}"))?;
        end = end.max(offset + source.length);
    }
    Ok(end)
}

fn write_bundle(plan: &BuildPlan, output_dir: &Path, header: BundleHeader, metadata_bytes: &[u8]) -> Result<u64> {
    fs::create_dir_all(output_dir)
        .with_context(|| format!("failed to create {}", output_dir.display()))?;
    let model_path = output_dir.join(&plan.manifest.model_file);
    let mut file = File::create(&model_path)
        .with_context(|| format!("failed to create {}", model_path.display()))?;

    file.write_all(&encode_header(&header))?;
    file.write_all(metadata_bytes)?;
    write_zero_padding(&mut file, header.data_offset - (64 + metadata_bytes.len() as u64))?;

    write_tensor_region(&mut file, header.data_offset, &plan.metadata, &plan.tensor_sources)?;
    if header.outlier_offset > 0 {
        file.seek(SeekFrom::End(0))?;
        let current = file.stream_position()?;
        write_zero_padding(&mut file, header.outlier_offset - current)?;
        write_outlier_region(&mut file, header.outlier_offset, &plan.metadata, &plan.outlier_sources)?;
    }
    if header.codebook_offset > 0 {
        file.seek(SeekFrom::End(0))?;
        let current = file.stream_position()?;
        write_zero_padding(&mut file, header.codebook_offset - current)?;
        write_codebook_region(&mut file, header.codebook_offset, &plan.metadata, &plan.codebook_sources)?;
    }
    if let Some(dedup) = &plan.metadata.expert_dedup {
        if dedup.region_offset > 0 {
            file.seek(SeekFrom::End(0))?;
            let current = file.stream_position()?;
            write_zero_padding(&mut file, dedup.region_offset - current)?;
            write_expert_dedup_region(&mut file, dedup.region_offset, &plan.metadata, &plan.expert_dedup_sources)?;
        }
    }
    if header.speculative_offset > 0 {
        file.seek(SeekFrom::End(0))?;
        let current = file.stream_position()?;
        write_zero_padding(&mut file, header.speculative_offset - current)?;
        write_speculative_draft_region(&mut file, &plan.speculative_draft_source)?;
    }
    if let Some(lora) = &plan.metadata.lora {
        if lora.region_offset > 0 {
            file.seek(SeekFrom::End(0))?;
            let current = file.stream_position()?;
            write_zero_padding(&mut file, lora.region_offset - current)?;
            write_lora_region(&mut file, lora.region_offset, &plan.metadata, &plan.lora_sources)?;
        }
    }

    Ok(file.metadata()?.len())
}

fn write_tensor_region(
    file: &mut File,
    data_offset: u64,
    metadata: &Metadata,
    sources: &BTreeMap<String, ByteSource>,
) -> Result<()> {
    for tensor_name in ordered_tensor_names(metadata) {
        let descriptor = &metadata.tensors[&tensor_name];
        if descriptor.dedup_canonical.is_some() || is_lora_tensor(metadata, descriptor) {
            continue;
        }
        let source = &sources[&tensor_name];
        let absolute = data_offset + descriptor.data_offset;
        let current = file.stream_position()?;
        if absolute > current {
            write_zero_padding(file, absolute - current)?;
        }
        copy_source_slice(source, file)?;
    }
    Ok(())
}

fn write_expert_dedup_region(
    file: &mut File,
    base_offset: u64,
    metadata: &Metadata,
    sources: &BTreeMap<String, ByteSource>,
) -> Result<()> {
    for tensor_name in ordered_tensor_names(metadata) {
        let descriptor = &metadata.tensors[&tensor_name];
        let Some(relative) = descriptor.dedup_correction_offset else {
            continue;
        };
        let source = sources
            .get(&tensor_name)
            .ok_or_else(|| anyhow!("missing expert dedup source for {tensor_name}"))?;
        let absolute = base_offset + relative;
        let current = file.stream_position()?;
        if absolute > current {
            write_zero_padding(file, absolute - current)?;
        }
        copy_source_slice(source, file)?;
    }
    Ok(())
}

fn write_outlier_region(
    file: &mut File,
    base_offset: u64,
    metadata: &Metadata,
    sources: &BTreeMap<String, ByteSource>,
) -> Result<()> {
    for tensor_name in ordered_tensor_names(metadata) {
        let descriptor = &metadata.tensors[&tensor_name];
        let Some(relative) = descriptor.outlier_indices_offset else {
            continue;
        };
        let source = sources
            .get(&tensor_name)
            .ok_or_else(|| anyhow!("missing outlier source for {tensor_name}"))?;
        let absolute = base_offset + relative;
        let current = file.stream_position()?;
        if absolute > current {
            write_zero_padding(file, absolute - current)?;
        }
        copy_source_slice(source, file)?;
    }
    Ok(())
}

fn write_codebook_region(
    file: &mut File,
    base_offset: u64,
    metadata: &Metadata,
    sources: &BTreeMap<String, ByteSource>,
) -> Result<()> {
    for (codebook_id, descriptor) in &metadata.codebooks {
        let source = sources
            .get(codebook_id)
            .ok_or_else(|| anyhow!("missing codebook source for {codebook_id}"))?;
        let absolute = base_offset + descriptor.offset;
        let current = file.stream_position()?;
        if absolute > current {
            write_zero_padding(file, absolute - current)?;
        }
        copy_source_slice(source, file)?;
    }
    Ok(())
}

fn write_speculative_draft_region(file: &mut File, source: &Option<ByteSource>) -> Result<()> {
    let Some(source) = source else {
        bail!("speculative draft metadata is present but no draft source was provided");
    };
    copy_source_slice(source, file)
}

fn write_lora_region(
    file: &mut File,
    base_offset: u64,
    metadata: &Metadata,
    sources: &BTreeMap<String, ByteSource>,
) -> Result<()> {
    for tensor_name in ordered_tensor_names(metadata) {
        let descriptor = &metadata.tensors[&tensor_name];
        if !is_lora_tensor(metadata, descriptor) {
            continue;
        }
        let source = sources
            .get(&tensor_name)
            .ok_or_else(|| anyhow!("missing LoRA source for {tensor_name}"))?;
        let absolute = base_offset + descriptor.data_offset;
        let current = file.stream_position()?;
        if absolute > current {
            write_zero_padding(file, absolute - current)?;
        }
        copy_source_slice(source, file)?;
    }
    Ok(())
}

fn write_manifest_and_assets(plan: &BuildPlan, output_dir: &Path) -> Result<()> {
    for asset in &plan.assets {
        let destination = output_dir.join(&asset.dest);
        if let Some(parent) = destination.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::copy(&asset.source, &destination).with_context(|| {
            format!(
                "failed to copy asset {} to {}",
                asset.source,
                destination.display()
            )
        })?;
    }
    let manifest_path = output_dir.join("manifest.json");
    fs::write(&manifest_path, serde_json::to_string_pretty(&plan.manifest)?)
        .with_context(|| format!("failed to write {}", manifest_path.display()))?;
    Ok(())
}

fn build_header(
    flags: u32,
    header_len: u32,
    data_offset: u64,
    outlier_offset: u64,
    codebook_offset: u64,
    tail_offset: u64,
    boot_cutoff: u8,
    speculative_offset: u64,
) -> BundleHeader {
    BundleHeader {
        version_major: VERSION_MAJOR,
        version_minor: VERSION_MINOR,
        flags,
        header_len,
        data_offset,
        outlier_offset,
        codebook_offset,
        tail_offset,
        boot_cutoff,
        speculative_offset,
    }
}

fn encode_header(header: &BundleHeader) -> [u8; 64] {
    let mut bytes = [0_u8; 64];
    bytes[0..4].copy_from_slice(&MAGIC);
    bytes[4..6].copy_from_slice(&header.version_major.to_le_bytes());
    bytes[6..8].copy_from_slice(&header.version_minor.to_le_bytes());
    bytes[8..12].copy_from_slice(&header.flags.to_le_bytes());
    bytes[12..16].copy_from_slice(&header.header_len.to_le_bytes());
    bytes[16..24].copy_from_slice(&header.data_offset.to_le_bytes());
    bytes[24..32].copy_from_slice(&header.outlier_offset.to_le_bytes());
    bytes[32..40].copy_from_slice(&header.codebook_offset.to_le_bytes());
    bytes[40..48].copy_from_slice(&header.tail_offset.to_le_bytes());
    bytes[48] = header.boot_cutoff;
    bytes[52..60].copy_from_slice(&header.speculative_offset.to_le_bytes());
    bytes
}

fn read_header(file: &mut File) -> Result<BundleHeader> {
    let mut bytes = [0_u8; 64];
    file.read_exact(&mut bytes)?;
    if bytes[0..4] != MAGIC {
        bail!("invalid AXON magic");
    }
    if bytes[49..52].iter().any(|byte| *byte != 0) || bytes[60..64].iter().any(|byte| *byte != 0) {
        bail!("reserved AXON header bytes must be zero");
    }
    Ok(BundleHeader {
        version_major: u16::from_le_bytes(bytes[4..6].try_into().unwrap()),
        version_minor: u16::from_le_bytes(bytes[6..8].try_into().unwrap()),
        flags: u32::from_le_bytes(bytes[8..12].try_into().unwrap()),
        header_len: u32::from_le_bytes(bytes[12..16].try_into().unwrap()),
        data_offset: u64::from_le_bytes(bytes[16..24].try_into().unwrap()),
        outlier_offset: u64::from_le_bytes(bytes[24..32].try_into().unwrap()),
        codebook_offset: u64::from_le_bytes(bytes[32..40].try_into().unwrap()),
        tail_offset: u64::from_le_bytes(bytes[40..48].try_into().unwrap()),
        boot_cutoff: bytes[48],
        speculative_offset: u64::from_le_bytes(bytes[52..60].try_into().unwrap()),
    })
}

fn validate_header(header: &BundleHeader) -> Result<()> {
    if header.version_major > VERSION_MAJOR {
        bail!("unsupported AXON version {}.{}", header.version_major, header.version_minor);
    }
    if header.data_offset < 64 {
        bail!("data offset is invalid");
    }
    if header.flags & FLAG_HAS_BOOT_REGION == 0 {
        if header.tail_offset != 0 || header.boot_cutoff != 0 {
            bail!("tail_offset and boot_cutoff must be zero when HAS_BOOT_REGION is not set");
        }
    } else if header.tail_offset < header.data_offset {
        bail!("tail_offset must not precede data_offset");
    }
    if header.flags & FLAG_HAS_SPECULATIVE_DRAFT == 0 && header.speculative_offset != 0 {
        bail!("speculative_offset must be zero when HAS_SPECULATIVE_DRAFT is not set");
    }
    Ok(())
}

fn read_metadata_bytes(file: &mut File, header: &BundleHeader) -> Result<Vec<u8>> {
    file.seek(SeekFrom::Start(64))?;
    let mut metadata_bytes = vec![0_u8; header.header_len as usize];
    file.read_exact(&mut metadata_bytes)?;
    if header.flags & FLAG_HEADER_ZSTD != 0 {
        return zstd::stream::decode_all(&metadata_bytes[..]).context("failed to decompress metadata");
    }
    Ok(metadata_bytes)
}

fn validate_metadata(metadata: &Metadata, header: &BundleHeader) -> Result<()> {
    validate_metadata_shape(metadata)?;
    if !metadata
        .version
        .starts_with(&format!("{}.{}", header.version_major, header.version_minor))
    {
        bail!("metadata version {} does not match header", metadata.version);
    }
    if !metadata.codebooks.is_empty() && header.codebook_offset == 0 {
        bail!("metadata contains codebooks but header codebook offset is zero");
    }
    if metadata
        .tensors
        .values()
        .any(|tensor| tensor.outlier_indices_offset.is_some())
        && header.outlier_offset == 0
    {
        bail!("metadata references outliers but header outlier offset is zero");
    }
    if metadata.boot_region_bytes.unwrap_or(0) > 0 && header.tail_offset == 0 {
        bail!("metadata boot_region_bytes is set but header tail_offset is zero");
    }
    if metadata.speculative_draft.is_some() && header.speculative_offset == 0 {
        bail!("metadata speculative_draft is set but header speculative_offset is zero");
    }
    if let Some(lora) = &metadata.lora {
        if lora.region_offset == 0 {
            bail!("metadata lora.region_offset must be non-zero when lora metadata is present");
        }
    }
    Ok(())
}

fn validate_manifest_files(bundle_dir: &Path, manifest: &BundleManifest) -> Result<()> {
    let model_path = bundle_dir.join(&manifest.model_file);
    if !model_path.is_file() {
        bail!("missing model file {}", model_path.display());
    }
    if let Some(config_file) = &manifest.config_file {
        ensure_bundle_file(bundle_dir, config_file)?;
    }
    if let Some(generation_config_file) = &manifest.generation_config_file {
        ensure_bundle_file(bundle_dir, generation_config_file)?;
    }
    if let Some(tokenizer) = &manifest.tokenizer {
        for file in &tokenizer.files {
            ensure_bundle_file(bundle_dir, file)?;
        }
    }
    Ok(())
}

fn ensure_bundle_file(bundle_dir: &Path, file_name: &str) -> Result<()> {
    let path = bundle_dir.join(file_name);
    if !path.is_file() {
        bail!("missing bundle asset {}", path.display());
    }
    Ok(())
}

fn validate_tensor_regions(file: &mut File, header: &BundleHeader, metadata: &Metadata, file_size: u64) -> Result<()> {
    let mut ranges: Vec<(u64, u64, &str)> = Vec::new();
    for (name, tensor) in &metadata.tensors {
        if let Some(canonical) = &tensor.dedup_canonical {
            if !metadata.tensors.contains_key(canonical) {
                bail!("tensor {name} references missing dedup canonical {canonical}");
            }
            if tensor.dedup_correction_offset.is_none() || tensor.dedup_correction_count.is_none() {
                bail!("tensor {name} is missing dedup correction metadata");
            }
            continue;
        }
        let start = tensor_absolute_start(header, metadata, tensor)?;
        let end = start
            .checked_add(tensor.data_bytes)
            .ok_or_else(|| anyhow!("tensor region overflow for {name}"))?;
        if end > file_size {
            bail!("tensor {name} exceeds file bounds");
        }
        if let Some(codebook_id) = &tensor.codebook_id {
            if !metadata.codebooks.contains_key(codebook_id) {
                bail!("tensor {name} references missing codebook {codebook_id}");
            }
        }
        if let Some(scale_name) = &tensor.smoothquant_scale {
            if !metadata.smoothquant_scales.contains_key(scale_name) {
                bail!("tensor {name} references missing smoothquant scale {scale_name}");
            }
        }
        if let Some(priority) = tensor.prefetch_priority {
            if !(0.0..=1.0).contains(&priority) {
                bail!("tensor {name} has invalid prefetch_priority");
            }
        }
        if header.flags & FLAG_HAS_CHECKSUMS != 0 {
            let expected = tensor
                .checksum_xxh64
                .as_deref()
                .ok_or_else(|| anyhow!("tensor {name} is missing checksum_xxh64"))?;
            let actual = checksum_file_region(file, start, tensor.data_bytes)?;
            if actual != expected {
                bail!("checksum mismatch for tensor {name}");
            }
        }
        ranges.push((start, end, name.as_str()));
    }
    ranges.sort_by_key(|range| range.0);
    for window in ranges.windows(2) {
        if window[0].1 > window[1].0 {
            bail!("tensor regions overlap: {} and {}", window[0].2, window[1].2);
        }
    }
    Ok(())
}

fn validate_outlier_regions(header: &BundleHeader, metadata: &Metadata, file_size: u64) -> Result<()> {
    for (name, tensor) in &metadata.tensors {
        let Some(offset) = tensor.outlier_indices_offset else {
            continue;
        };
        let outlier_count = tensor
            .outlier_count
            .ok_or_else(|| anyhow!("tensor {name} is missing outlier_count"))?;
        let rows = tensor.shape.first().copied().unwrap_or(0);
        let bytes = (rows + 1) * 4 + outlier_count * 4 + outlier_count * 2;
        let start = header.outlier_offset + offset;
        let end = start
            .checked_add(bytes)
            .ok_or_else(|| anyhow!("outlier region overflow for {name}"))?;
        if end > file_size {
            bail!("outlier region for tensor {name} exceeds file bounds");
        }
    }
    Ok(())
}

fn validate_codebook_regions(header: &BundleHeader, metadata: &Metadata, file_size: u64) -> Result<()> {
    let mut ranges: Vec<(u64, u64, &str)> = Vec::new();
    for (name, codebook) in &metadata.codebooks {
        let start = header.codebook_offset + codebook.offset;
        let end = start
            .checked_add(codebook.size)
            .ok_or_else(|| anyhow!("codebook region overflow for {name}"))?;
        if end > file_size {
            bail!("codebook {name} exceeds file bounds");
        }
        ranges.push((start, end, name.as_str()));
    }
    for (name, scale) in &metadata.smoothquant_scales {
        let start = header.codebook_offset + scale.offset;
        let end = start
            .checked_add(scale.size)
            .ok_or_else(|| anyhow!("smoothquant region overflow for {name}"))?;
        if end > file_size {
            bail!("smoothquant scale {name} exceeds file bounds");
        }
        ranges.push((start, end, name.as_str()));
    }
    ranges.sort_by_key(|range| range.0);
    for window in ranges.windows(2) {
        if window[0].1 > window[1].0 {
            bail!("codebook regions overlap: {} and {}", window[0].2, window[1].2);
        }
    }
    Ok(())
}

fn validate_expert_dedup_regions(metadata: &Metadata, file_size: u64) -> Result<()> {
    let Some(dedup) = &metadata.expert_dedup else {
        return Ok(());
    };
    let mut ranges: Vec<(u64, u64, &str)> = Vec::new();
    for (name, tensor) in &metadata.tensors {
        let Some(offset) = tensor.dedup_correction_offset else {
            continue;
        };
        let count = tensor
            .dedup_correction_count
            .ok_or_else(|| anyhow!("tensor {name} is missing dedup_correction_count"))?;
        let rows = tensor.shape.first().copied().unwrap_or(0);
        let bytes = (rows + 1) * 4 + count * 4 + count * 2;
        let start = dedup.region_offset + offset;
        let end = start
            .checked_add(bytes)
            .ok_or_else(|| anyhow!("expert dedup region overflow for {name}"))?;
        if end > file_size {
            bail!("expert dedup region for tensor {name} exceeds file bounds");
        }
        ranges.push((start, end, name.as_str()));
    }
    ranges.sort_by_key(|range| range.0);
    for window in ranges.windows(2) {
        if window[0].1 > window[1].0 {
            bail!("expert dedup regions overlap: {} and {}", window[0].2, window[1].2);
        }
    }
    Ok(())
}

fn validate_speculative_region(header: &BundleHeader, metadata: &Metadata, file_size: u64) -> Result<()> {
    let Some(speculative) = &metadata.speculative_draft else {
        return Ok(());
    };
    let start = header
        .speculative_offset
        .checked_add(speculative.draft_offset)
        .ok_or_else(|| anyhow!("speculative draft offset overflow"))?;
    let end = start
        .checked_add(speculative.draft_bytes)
        .ok_or_else(|| anyhow!("speculative draft size overflow"))?;
    if end > file_size {
        bail!("speculative draft region exceeds file bounds");
    }
    Ok(())
}

fn tensor_absolute_start(header: &BundleHeader, metadata: &Metadata, tensor: &TensorDescriptor) -> Result<u64> {
    if is_lora_tensor(metadata, tensor) {
        let lora = metadata
            .lora
            .as_ref()
            .ok_or_else(|| anyhow!("LoRA tensor is present but metadata.lora is missing"))?;
        Ok(lora.region_offset + tensor.data_offset)
    } else {
        Ok(header.data_offset + tensor.data_offset)
    }
}

fn checksum_file_region(file: &mut File, start: u64, length: u64) -> Result<String> {
    file.seek(SeekFrom::Start(start))?;
    let mut remaining = length;
    let mut buffer = vec![0_u8; 64 * 1024];
    let mut hasher = Xxh64::new(0);
    while remaining > 0 {
        let read_len = remaining.min(buffer.len() as u64) as usize;
        file.read_exact(&mut buffer[..read_len])?;
        hasher.update(&buffer[..read_len]);
        remaining -= read_len as u64;
    }
    Ok(format!("{:016x}", hasher.digest()))
}

fn copy_source_slice(source: &ByteSource, output: &mut File) -> Result<()> {
    let mut input = File::open(&source.path).with_context(|| format!("failed to open {}", source.path))?;
    input.seek(SeekFrom::Start(source.offset))?;
    let mut remaining = source.length;
    let mut buffer = vec![0_u8; 64 * 1024];
    while remaining > 0 {
        let read_len = remaining.min(buffer.len() as u64) as usize;
        input.read_exact(&mut buffer[..read_len])?;
        output.write_all(&buffer[..read_len])?;
        remaining -= read_len as u64;
    }
    Ok(())
}

fn write_zero_padding(file: &mut File, count: u64) -> Result<()> {
    if count == 0 {
        return Ok(());
    }
    let zeros = vec![0_u8; 4096];
    let mut remaining = count;
    while remaining > 0 {
        let write_len = remaining.min(zeros.len() as u64) as usize;
        file.write_all(&zeros[..write_len])?;
        remaining -= write_len as u64;
    }
    Ok(())
}

fn align64(value: u64) -> u64 {
    if value % 64 == 0 {
        value
    } else {
        value + (64 - (value % 64))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn align64_rounds_up() {
        assert_eq!(align64(0), 0);
        assert_eq!(align64(1), 64);
        assert_eq!(align64(64), 64);
        assert_eq!(align64(65), 128);
    }

    #[test]
    fn header_round_trip() {
        let header = build_header(FLAG_HAS_CHECKSUMS, 128, 256, 512, 768, 0, 0, 0);
        let encoded = encode_header(&header);
        let temp_dir = tempdir().unwrap();
        let path = temp_dir.path().join("header.axon");
        fs::write(&path, encoded).unwrap();
        let mut file = File::open(path).unwrap();
        let decoded = read_header(&mut file).unwrap();
        assert_eq!(decoded.flags, header.flags);
        assert_eq!(decoded.header_len, header.header_len);
        assert_eq!(decoded.data_offset, header.data_offset);
        assert_eq!(decoded.outlier_offset, header.outlier_offset);
        assert_eq!(decoded.codebook_offset, header.codebook_offset);
    }
}

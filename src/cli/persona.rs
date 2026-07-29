//! `plakat persona` — controllable synthetic-person composition (RFC PERSONA-1, the 5.0.0 flagship).
//!
//! P0 first slice: `new` (scaffold a spec) + `lint` (validate without weights). The resolver, geometry,
//! details, casting, scorecard and TUI subcommands land across later phases (ROADMAP_5.0.0). Fully
//! additive — nothing here touches existing behaviour.

use anyhow::{Context, Result};
use clap::{Args, Subcommand};
use console::style;
use std::path::PathBuf;

use crate::persona::lint::{self, Level};
use crate::persona::PersonaSpec;

#[derive(Args, Debug)]
pub struct PersonaArgs {
    #[command(subcommand)]
    pub cmd: PersonaCmd,
}

#[derive(Subcommand, Debug)]
pub enum PersonaCmd {
    /// Scaffold a new persona spec (a valid, partial `PersonaSpec` HJSON you then edit or `--tui`).
    New(NewArgs),
    /// Validate a persona spec — schema, scalar ranges, contradictions, the age gate. No weights, no
    /// network. Exits non-zero on any error so it can gate CI.
    Lint(LintArgs),
    /// Show what a spec resolves to for a model family: the salience-ranked prompt-routed attributes,
    /// the compiled positive/negative prompt, and (on CLIP) which attributes the token budget dropped.
    Show(ShowArgs),
    /// Measure a rendered image against a spec (the scorecard, RFC §12). P1: the landmark probe —
    /// SCRFD detects the face, PIPNet-98 aligns it, and geometric ratio metrics are reported.
    Verify(VerifyArgs),
    /// Rasterise the geometry maps a spec resolves to (the Layer-2 geometry engine, RFC §10) — mesh /
    /// wireframe / depth / pose-skeleton / region-masks / dentition / figure. Pure, no weights.
    Geometry(GeometryArgs),
    /// Composite a spec's localized details (marks / jewelry) onto a rendered image (RFC §8) —
    /// anchored through the realised landmarks, deterministic. `--harmonise` blends them into skin.
    Composite(CompositeArgs),
}

#[derive(Args, Debug)]
pub struct NewArgs {
    /// Output path for the new spec (`.hjson`).
    pub out: PathBuf,
    /// How much of the schema to scaffold: `quick` (identity + core structure), `standard`, `full`.
    #[arg(long, default_value = "quick")]
    pub depth: String,
    /// Slug name for the persona (also the People-library folder name).
    #[arg(long, default_value = "unnamed")]
    pub name: String,
    /// Apparent age in years (must be >= 18; see §23.1).
    #[arg(long, default_value_t = 30)]
    pub age: u32,
}

#[derive(Args, Debug)]
pub struct LintArgs {
    /// Path to the persona spec (`.hjson`).
    pub spec: PathBuf,
}

#[derive(Args, Debug)]
pub struct ShowArgs {
    /// Path to the persona spec (`.hjson`).
    pub spec: PathBuf,
    /// Model family to compile for (its encoder class shapes the prompt). Default `sdxl`.
    #[arg(long, default_value = "sdxl")]
    pub model: String,
}

#[derive(Args, Debug)]
pub struct VerifyArgs {
    /// Path to the persona spec (`.hjson`).
    pub spec: PathBuf,
    /// The rendered image to measure against the spec.
    #[arg(long)]
    pub image: PathBuf,
}

#[derive(Args, Debug)]
pub struct GeometryArgs {
    /// Path to the persona spec (`.hjson`).
    pub spec: PathBuf,
    /// Directory to write the map PNG(s) into (created if missing).
    #[arg(long, default_value = "persona-geometry")]
    pub out: PathBuf,
    /// Which map(s): `all` or a comma list of
    /// mesh,wireframe,depth,skeleton,masks,dentition,figure.
    #[arg(long, default_value = "all")]
    pub map: String,
    /// Output edge length in pixels (face maps are square; the figure map is 3:4).
    #[arg(long, default_value_t = 512)]
    pub size: u32,
    /// Mesh drawing convention: `mediapipe` (per-feature colour) or `generic` (white).
    #[arg(long = "mesh-style", default_value = "mediapipe")]
    pub mesh_style: String,
    /// Seed for the asymmetry perturbation (§10.2).
    #[arg(long, default_value_t = 0)]
    pub seed: u64,
}

#[derive(Args, Debug)]
pub struct CompositeArgs {
    /// Path to the persona spec (`.hjson`).
    pub spec: PathBuf,
    /// The rendered image to composite the details onto.
    #[arg(long)]
    pub image: PathBuf,
    /// Output path for the composited image.
    #[arg(long, default_value = "composited.png")]
    pub out: PathBuf,
    /// Also write the union affected-region mask here (for inspection / external harmonisation).
    #[arg(long)]
    pub mask: Option<PathBuf>,
    /// Seed for the procedural overlays (§8.3).
    #[arg(long, default_value_t = 0)]
    pub seed: u64,
    /// Run the harmonisation img2img pass (§8.4) over the affected region to blend the overlays into
    /// skin. Requires model weights; the deterministic composite runs regardless.
    #[arg(long)]
    pub harmonise: bool,
    /// Model for the harmonisation pass.
    #[arg(long, default_value = "sdxl")]
    pub model: String,
    /// Harmonisation denoise strength — low, so the overlays are blended, not regenerated.
    #[arg(long = "harmonise-strength", default_value_t = 0.25)]
    pub harmonise_strength: f32,
}

pub async fn run(args: PersonaArgs) -> Result<()> {
    match args.cmd {
        PersonaCmd::New(a) => run_new(a),
        PersonaCmd::Lint(a) => run_lint(a),
        PersonaCmd::Show(a) => run_show(a),
        PersonaCmd::Verify(a) => run_verify(a).await,
        PersonaCmd::Geometry(a) => run_geometry(a),
        PersonaCmd::Composite(a) => run_composite(a).await,
    }
}

async fn run_composite(a: CompositeArgs) -> Result<()> {
    use crate::persona::{detail, scorecard};
    let spec = PersonaSpec::load(&a.spec)?;
    let n_marks = spec.marks.as_ref().map(|m| m.len()).unwrap_or(0);
    let n_piercings = spec.piercings.as_ref().map(|p| p.len()).unwrap_or(0);
    let n_jewelry = spec.jewelry.as_ref().and_then(|j| j.items.as_ref()).map(|i| i.len()).unwrap_or(0);
    let n_details = n_marks + n_piercings + n_jewelry;
    if n_details == 0 {
        println!("{}  spec has no marks / piercings / jewelry — nothing to composite", style("·").dim());
        // still a no-op copy so downstream steps have an output.
        image::open(&a.image)?.to_rgb8().save(&a.out)?;
        println!("  {} {}", style("✓").green(), a.out.display());
        return Ok(());
    }

    // Detect + align the rendered face (the realised landmarks the anchors resolve through, §8.4).
    let device = candle_core::Device::Cpu;
    let (detector, pipnet) = scorecard::load_probes(&device).await?;
    let Some(m) = scorecard::measure_landmarks(&a.image, &detector, &pipnet)? else {
        println!("{}  no face detected in {} — cannot anchor details", style("✗").red(), a.image.display());
        return Ok(());
    };

    let base = image::open(&a.image)?.to_rgb8();
    let r = detail::composite_details(&base, &spec, &m, a.seed);
    println!(
        "{}  {}  (face score {:.2})",
        style("persona composite").bold(),
        a.image.display(),
        m.detection_score
    );
    println!(
        "  placed {} / {n_details} detail(s); light ({:+.2}, {:+.2})",
        r.placed,
        r.light.dx,
        r.light.dy
    );
    for c in &r.culled {
        println!("  {} culled {}: {}", style("·").yellow(), c.kind, c.reason);
    }

    r.image.save(&a.out).with_context(|| format!("writing {}", a.out.display()))?;
    println!("  {} {}", style("✓").green(), a.out.display());
    if let Some(mp) = &a.mask {
        r.mask.save(mp).with_context(|| format!("writing {}", mp.display()))?;
        println!("  {} {} (affected-region mask)", style("✓").green(), mp.display());
    }

    // Optional harmonisation: low-strength masked img2img over the affected region (§8.4).
    if a.harmonise {
        use crate::api::Img2img;
        let tmp = std::env::temp_dir();
        let comp_path = tmp.join(format!("persona_harm_{}.png", a.seed));
        let mask_path = tmp.join(format!("persona_harm_mask_{}.png", a.seed));
        r.image.save(&comp_path)?;
        r.mask.save(&mask_path)?;
        println!("  {} harmonising over the affected region (model {}, strength {:.2})…", style("→").cyan(), a.model, a.harmonise_strength);
        let prompt = "a portrait photograph, natural skin, detailed skin texture";
        let out = Img2img::new(&a.model, &comp_path)
            .prompt(prompt)
            .mask(&mask_path)
            .mask_feather(6)
            .strength(a.harmonise_strength)
            .seed(a.seed)
            .run()
            .await?;
        if let Some(img) = out.into_iter().next() {
            img.save(&a.out)?;
            println!("  {} {} (harmonised)", style("✓").green(), a.out.display());
        }
        let _ = std::fs::remove_file(&comp_path);
        let _ = std::fs::remove_file(&mask_path);
    }
    Ok(())
}

fn run_geometry(a: GeometryArgs) -> Result<()> {
    use crate::persona::geometry as geo;
    let spec = PersonaSpec::load(&a.spec)?;
    let sz = a.size.clamp(64, 4096);
    let mesh_style = match a.mesh_style.as_str() {
        "generic" => geo::MeshStyle::Generic,
        _ => geo::MeshStyle::MediaPipe,
    };

    // resolve the face geometry from the spec.
    let values = geo::geometry_values(&spec);
    let open = geo::open_mouth(&spec);
    let d = geo::resolve(&values, open, a.seed);

    std::fs::create_dir_all(&a.out).with_context(|| format!("creating {}", a.out.display()))?;
    let want: Vec<&str> = if a.map == "all" {
        vec!["mesh", "wireframe", "depth", "skeleton", "masks", "dentition", "figure"]
    } else {
        a.map.split(',').map(|s| s.trim()).filter(|s| !s.is_empty()).collect()
    };

    let name = spec.identity.as_ref().and_then(|i| i.name.as_deref()).unwrap_or("persona");
    println!("{}  {}  (seed {}, {}²)", style("persona geometry").bold(), name, a.seed, sz);
    if !d.warnings.is_empty() {
        for w in &d.warnings {
            println!("  {} {}: {}", style("validity").yellow(), w.kind, w.detail);
        }
    }

    let mut wrote = 0;
    let save = |path: PathBuf, r: image::DynamicImage| -> Result<()> {
        r.save(&path).with_context(|| format!("writing {}", path.display()))?;
        println!("  {} {}", style("✓").green(), path.display());
        Ok(())
    };
    for m in &want {
        let p = |suffix: &str| a.out.join(format!("{name}_{suffix}.png"));
        match *m {
            "mesh" => {
                save(p("mesh"), geo::mesh_map(&d.landmarks, sz, mesh_style).into())?;
                wrote += 1;
            }
            "wireframe" | "wire" => {
                save(p("wireframe"), geo::wireframe(&d.landmarks, sz).into())?;
                wrote += 1;
            }
            "depth" => {
                save(p("depth"), geo::depth_proxy(&d.landmarks, sz).into())?;
                wrote += 1;
            }
            "skeleton" | "pose" => {
                save(p("skeleton"), geo::face_skeleton(&d.landmarks, sz).into())?;
                wrote += 1;
            }
            "masks" | "mask" => {
                // composite the region masks into one colour-coded image.
                let mut img = image::RgbImage::new(sz, sz);
                for (r, c) in [
                    (geo::Region::Face, [40u8, 40, 60]),
                    (geo::Region::BrowRight, [200, 160, 60]),
                    (geo::Region::BrowLeft, [200, 160, 60]),
                    (geo::Region::EyeRight, [80, 200, 255]),
                    (geo::Region::EyeLeft, [80, 200, 255]),
                    (geo::Region::Nose, [120, 255, 120]),
                    (geo::Region::Mouth, [255, 90, 90]),
                ] {
                    let mask = geo::region_mask(&d.landmarks, sz, r);
                    for (x, y, px) in mask.enumerate_pixels() {
                        if px.0[0] > 0 {
                            img.put_pixel(x, y, image::Rgb(c));
                        }
                    }
                }
                save(p("masks"), img.into())?;
                wrote += 1;
            }
            "dentition" | "teeth" => {
                save(p("dentition"), geo::dentition_hint(&d.landmarks, sz, 6).into())?;
                wrote += 1;
            }
            "figure" => match geo::figure_params(&spec) {
                Some(fp) => {
                    let (w, h) = (sz, sz * 4 / 3);
                    let fig = geo::resolve_figure(&fp, a.seed);
                    save(p("figure_skeleton"), geo::figure_skeleton_map(&fig, w, h).into())?;
                    save(p("figure_silhouette"), geo::silhouette_mask(&fig, w, h).into())?;
                    wrote += 2;
                }
                None => {
                    if a.map != "all" {
                        println!("  {} figure: spec has no `figure:` block — nothing to render", style("·").dim());
                    }
                }
            },
            other => println!("  {} unknown map `{other}` (skipped)", style("?").yellow()),
        }
    }
    println!("{} {wrote} map(s) → {}", style("done:").bold(), a.out.display());
    Ok(())
}

async fn run_verify(a: VerifyArgs) -> Result<()> {
    use crate::persona::scorecard::{self, AttrScore, DetailSubscore, Scorecard};
    use crate::persona::spec::Color;
    let spec = PersonaSpec::load(&a.spec)?;
    let lex = crate::persona::lexicon::Lexicon::skeleton();
    let weight = |p: &str| lex.get(p).map(|e| e.control_weight() * e.class_weight()).unwrap_or(0.7);
    let mut sc = Scorecard::default();

    // Measurement runs on CPU (the aligner is tiny + deterministic; avoids GPU contention).
    let device = candle_core::Device::Cpu;
    let (detector, pipnet) = scorecard::load_probes(&device).await?;
    let Some(m) = scorecard::measure_landmarks(&a.image, &detector, &pipnet)? else {
        println!("{}  no face detected in {}", style("✗").red(), a.image.display());
        return Ok(());
    };
    println!("{}  {}  (face score {:.2})", style("scorecard").bold(), a.image.display(), m.detection_score);

    // --- landmark metrics (scalars → pending calibration until the P4 prior exists) ---
    println!("\n{}", style("landmark metrics (WFLW-98):").bold());
    println!("  interpupillary / face-width   {:.3}   ({})", m.interpupillary_over_facewidth, style("eyes.spacing").dim());
    println!("  mouth-width / face-width      {:.3}   ({})", m.mouth_over_facewidth, style("mouth.width").dim());
    println!("  face aspect (h/w)             {:.3}   ({})", m.face_aspect, style("face.width⁻¹").dim());
    for (path, set) in [
        ("eyes.spacing", spec.eyes.as_ref().and_then(|e| e.spacing).is_some()),
        ("mouth.width", spec.mouth.as_ref().and_then(|mo| mo.width).is_some()),
        ("face.width", spec.face.as_ref().and_then(|f| f.width).is_some()),
    ] {
        if set {
            sc.pending_calibration.push(path.to_string());
        }
    }

    // --- region_color (eyes.color scorable now via ΔE; skin reported) ---
    let colors = scorecard::measure_colors(&m);
    println!("\n{}", style("region colour (CIELAB):").bold());
    let fmt = |l: Option<[f32; 3]>| l.map(|v| format!("L{:.0} a{:+.0} b{:+.0}", v[0], v[1], v[2])).unwrap_or_else(|| "—".into());
    let eye_target = spec.eyes.as_ref().and_then(|e| e.color.as_ref()).and_then(|c| match c {
        Color::Named(n) => scorecard::color_name_to_lab(n),
        Color::Lab { lab } => Some(*lab),
    });
    match (colors.iris, eye_target) {
        (Some(iris), Some(t)) => {
            let de = scorecard::delta_e(iris, t);
            let pass = de < 20.0;
            println!("  iris  {}   → ΔE {:.1} to eyes.color   {}", fmt(Some(iris)), de, if pass { style("✓").green() } else { style("✗").red() });
            sc.scored.push(AttrScore { path: "eyes.color".into(), pass, weight: weight("eyes.color"), note: format!("ΔE {de:.1}") });
        }
        (iris, _) => {
            println!("  iris  {}   ({})", fmt(iris), style("eyes.color").dim());
            if spec.eyes.as_ref().and_then(|e| e.color.as_ref()).is_some() {
                sc.unmeasurable.push("eyes.color".into());
            }
        }
    }
    println!("  skin  {}   ({})", fmt(colors.skin), style("skin.tone").dim());

    // --- detail sub-score (marks via local_anomaly) ---
    let marks = spec.marks.as_deref().unwrap_or(&[]);
    let positional: Vec<_> = marks.iter().filter(|mk| mk.anchor.is_some()).collect();
    if !positional.is_empty() {
        println!("\n{}", style("localized details (local_anomaly):").bold());
        let (mut pres_sum, mut pos_sum, mut n) = (0.0f32, 0.0f32, 0usize);
        for (i, mk) in positional.iter().enumerate() {
            let anchor = mk.anchor.as_ref().unwrap();
            let kind = mk.kind.as_deref().unwrap_or("mark");
            let name = anchor.landmark.as_deref().or(anchor.region.as_deref()).unwrap_or("?");
            let Some(pos) = scorecard::resolve_anchor(anchor, &m) else {
                println!("  {} {kind}[{i}]: anchor {name:?} not in the WFLW-98 vocabulary", style("?").yellow());
                continue;
            };
            let radius = mk.size.unwrap_or(0.05) * m.face_w;
            let r = scorecard::local_anomaly(&m.crop, pos, radius.max(0.02));
            let glyph = if r.presence > 0.5 { style("✓").green() } else { style("✗").red() };
            println!("  {glyph} {kind}[{i}] @ {}  presence {:.2}  position-error {:.3}", style(name).dim(), r.presence, r.position_error);
            pres_sum += r.presence;
            pos_sum += r.position_error;
            n += 1;
        }
        if n > 0 {
            sc.detail = Some(DetailSubscore { presence_mean: pres_sum / n as f32, position_mean: pos_sum / n as f32, n });
        }
    }

    // --- detect (OWL-ViT, lazy): facial hair + glasses (scorable present/absent) ---
    let mut queries: Vec<(String, String, String, bool)> = Vec::new(); // (label, path, query, expected)
    if let Some(style) = spec.facial_hair.as_ref().and_then(|f| f.style.as_deref()) {
        queries.push((format!("facial_hair.style={style}"), "facial_hair.style".into(), scorecard::facial_hair_query(style).into(), style != "none"));
    }
    if spec.jewelry.as_ref().and_then(|j| j.items.as_ref()).is_some_and(|it| it.iter().any(|x| x.kind.as_deref() == Some("glasses"))) {
        queries.push(("jewelry: glasses".into(), "jewelry.glasses".into(), "glasses".into(), true));
    }
    if !queries.is_empty() {
        let owl = crate::pipelines::owlvit::OwlViT::load_pretrained(&device).await?;
        println!("\n{}", style("salient objects (detect / OWL-ViT):").bold());
        for (label, path, query, expected) in &queries {
            let r = scorecard::detect_probe(&owl, &a.image, query, 0.1)?;
            let pass = r.present == *expected;
            let glyph = if pass { style("✓").green() } else { style("✗").red() };
            println!("  {glyph} {label}: {} (score {:.2}, expected {})", if r.present { "present" } else { "absent" }, r.score, if *expected { "present" } else { "absent" });
            sc.scored.push(AttrScore { path: path.clone(), pass, weight: weight(path), note: String::new() });
        }
    }

    // --- aggregate (§12.2): weighted pass fraction over scored attrs, exclusions reported separately ---
    println!("\n{}", style("── scorecard ──").bold());
    match sc.aggregate() {
        Some(agg) => {
            let styled = if agg >= 0.8 { style(format!("{agg:.2}")).green() } else if agg >= 0.5 { style(format!("{agg:.2}")).yellow() } else { style(format!("{agg:.2}")).red() };
            let passed = sc.scored.iter().filter(|s| s.pass).count();
            println!("  score {styled}  ({}/{} scored attributes pass)", passed, sc.scored.len());
        }
        None => println!("  score {}  (no attributes scorable yet on this spec)", style("—").dim()),
    }
    if let Some(d) = &sc.detail {
        println!("  detail sub-score: presence {:.2}, position {:.3} over {} mark(s)", d.presence_mean, d.position_mean, d.n);
    }
    let ex = |label: &str, v: &[String]| {
        if !v.is_empty() {
            println!("  {} {}: {}", style("·").dim(), style(label).dim(), v.join(", "));
        }
    };
    ex("pending calibration (P4)", &sc.pending_calibration);
    ex("unmeasurable", &sc.unmeasurable);
    ex("non-manifesting", &sc.non_manifesting);
    Ok(())
}

fn run_show(a: ShowArgs) -> Result<()> {
    use crate::persona::compile::{self, EncoderClass};
    use crate::persona::lexicon::Lexicon;
    let spec = PersonaSpec::load(&a.spec)?;
    let lex = Lexicon::skeleton();
    let resolved = compile::resolve(&spec, &lex);
    let compiled = compile::compile_for_model(&spec, &lex, &a.model);

    println!("{}  {}  (model {}, encoder {})", style("persona").bold(), a.spec.display(), a.model, style(compiled.class).cyan());
    let _ = EncoderClass::from_model(&a.model);
    println!("\n{}", style("resolved attributes (salience-ranked):").bold());
    if resolved.is_empty() {
        println!("  (none — spec is empty or all-unknown)");
    }
    for r in &resolved {
        let emitted = compiled.emitted.iter().any(|p| p == &r.path);
        let mark = if r.phrase.is_empty() {
            style("neg").yellow()
        } else if emitted {
            style("✓").green()
        } else {
            style("dropped").red()
        };
        let phrase = if r.phrase.is_empty() { "(negative only)" } else { &r.phrase };
        println!("  {mark} {:<22} sal {:.2}  {}", style(&r.path).dim(), r.salience, phrase);
    }
    if !compiled.dropped.is_empty() {
        println!("\n{} {}", style("dropped by budget:").red(), compiled.dropped.join(", "));
    }
    println!("\n{}\n  {}", style("positive:").bold().green(), if compiled.positive.is_empty() { "(empty)" } else { &compiled.positive });
    println!("{}\n  {}", style("negative:").bold().yellow(), if compiled.negative.is_empty() { "(empty)" } else { &compiled.negative });
    Ok(())
}

fn run_new(a: NewArgs) -> Result<()> {
    if a.out.exists() {
        anyhow::bail!("{} already exists — refusing to overwrite", a.out.display());
    }
    let body = scaffold(&a.depth, &a.name, a.age)?;
    // Sanity: the scaffold must itself load + lint clean.
    let spec = PersonaSpec::from_hjson(&body)
        .with_context(|| "internal: generated scaffold did not parse")?;
    let findings = lint::lint(&spec);
    if lint::has_errors(&findings) {
        anyhow::bail!("internal: generated scaffold did not lint clean: {findings:?}");
    }
    if let Some(parent) = a.out.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).ok();
        }
    }
    std::fs::write(&a.out, body).with_context(|| format!("writing {}", a.out.display()))?;
    println!("{}  persona scaffold ({})  →  {}", style("✓").green(), a.depth, a.out.display());
    println!("   edit it, or run `plakat persona lint {}`", a.out.display());
    Ok(())
}

fn run_lint(a: LintArgs) -> Result<()> {
    let spec = PersonaSpec::load(&a.spec)?;
    let findings = lint::lint(&spec);
    let (mut errs, mut warns, mut infos) = (0u32, 0u32, 0u32);
    for f in &findings {
        let (tag, styled) = match f.level {
            Level::Error => {
                errs += 1;
                ("error", style("error").red().bold())
            }
            Level::Warning => {
                warns += 1;
                ("warn", style("warn").yellow())
            }
            Level::Info => {
                infos += 1;
                ("info", style("info").cyan())
            }
        };
        let _ = tag;
        println!("  {styled} {}: {}", style(&f.path).bold(), f.message);
    }
    if findings.is_empty() {
        println!("{}  {} — clean", style("✓").green(), a.spec.display());
    } else {
        println!(
            "{}  {errs} error(s), {warns} warning(s), {infos} info",
            if errs > 0 { style("✗").red() } else { style("✓").green() },
        );
    }
    if errs > 0 {
        std::process::exit(1);
    }
    Ok(())
}

/// Build a valid, partial persona scaffold at the requested depth. HJSON is brace-based; quoteless
/// strings run to end-of-line, so every value sits alone on its line.
fn scaffold(depth: &str, name: &str, age: u32) -> Result<String> {
    let mut s = String::new();
    s.push_str("{\n  schema: persona/1\n\n");
    s.push_str("  identity: {\n");
    s.push_str(&format!("    name: {name}\n"));
    s.push_str(&format!("    apparent_age: {age}\n"));
    s.push_str("    sex: androgynous          # female | male | androgynous\n");
    s.push_str("  }\n\n");
    s.push_str("  # Scalars are 0..1 where 0.5 is the model's own prior; leave a field OUT to mean\n");
    s.push_str("  # \"unknown\" (an explicit 0.5 asserts \"deliberately average\"). See PERSONA.md.\n\n");
    s.push_str("  face: {\n    # shape: oval             # oval|round|square|heart|diamond|oblong|triangular\n");
    s.push_str("    # width: 0.5\n  }\n\n");
    s.push_str("  eyes: {\n    # color: hazel\n    # spacing: 0.5\n  }\n");

    if depth == "standard" || depth == "full" {
        s.push_str("\n  nose: {\n    # profile: straight\n  }\n");
        s.push_str("\n  mouth: {\n    # width: 0.5\n  }\n");
        s.push_str("\n  skin: {\n    # tone: fitzpatrick-3     # fitzpatrick-1..6\n  }\n");
        s.push_str("\n  hair: {\n    # color: auburn\n    # length: shoulder        # buzz|crop|short|chin|shoulder|long|very-long\n  }\n");
        s.push_str("\n  figure: {\n    # height_cm: 170\n    # build: mesomorph\n  }\n");
    }
    if depth == "full" {
        s.push_str("\n  teeth: {\n    # visibility: auto\n    # alignment: even\n  }\n");
        // Collections: an empty list ASSERTS \"none\"; omit the key entirely to mean \"undecided\".
        s.push_str("\n  # marks: []                 # [] asserts \"no marks\"; omit for \"undecided\"\n");
        s.push_str("  # piercings: []\n");
        s.push_str("\n  jewelry: {\n    # identity_locked: false\n    # items: []\n  }\n");
        s.push_str("\n  defaults: {\n    # expression: neutral\n    # framing: headshot\n  }\n");
    }
    if !matches!(depth, "quick" | "standard" | "full") {
        anyhow::bail!("unknown --depth {depth:?} (quick|standard|full)");
    }
    s.push_str("\n  provenance: {\n    method: manual\n    lexicon_version: \"1.0\"\n  }\n}\n");
    Ok(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scaffolds_lint_clean_at_every_depth() {
        for depth in ["quick", "standard", "full"] {
            let body = scaffold(depth, "alice", 34).unwrap();
            let spec = PersonaSpec::from_hjson(&body)
                .unwrap_or_else(|e| panic!("depth {depth} did not parse: {e}\n{body}"));
            let f = lint::lint(&spec);
            assert!(!lint::has_errors(&f), "depth {depth} did not lint clean: {f:?}");
            assert_eq!(spec.schema_version(), Some(1));
            assert_eq!(spec.identity.unwrap().apparent_age, Some(34));
        }
    }

    #[test]
    fn scaffold_rejects_bad_depth() {
        assert!(scaffold("deep", "x", 30).is_err());
    }
}

// 이미지 파일을 선택해서 넣으면 YOLOv8, Gemini를 거쳐 json 파일을 자동으로 저장하는 사용자 친화적 프로그램
// I/O 작업에서의 빠른 속도를 위해 Rust로 만듦.
use anyhow::{Context, Result};
use chrono::Local;
use eframe::{egui, egui::Color32};
use eframe::egui::Widget;
use egui_extras::{TableBuilder, Column};
use rfd::FileDialog;
use serde::Deserialize;
use std::{
    collections::{HashMap, HashSet},
    env, fs,
    path::{Path, PathBuf},
    process::Command,
};

#[derive(Debug, Deserialize, Clone)]
struct WheelResultFile { results: Vec<WheelOne> }
#[derive(Debug, Deserialize, Clone)]
struct WheelOne { image: String, result: WheelJudge }
#[derive(Debug, Deserialize, Clone)]
struct WheelJudge { accessible: Option<bool>, reason: String }

fn main() {
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size(egui::vec2(1024.0, 720.0))
            .with_min_inner_size(egui::vec2(900.0, 600.0)),
        ..Default::default()
    };
    if let Err(e) = eframe::run_native(
        "Wheel City AI 2 – Runner",
        native_options,
        Box::new(|_cc| Box::new(AppState::default())),
    ) {
        eprintln!("Failed to start app: {e:?}");
    }
}

struct AppState {
    // inputs
    pending_files: Vec<PathBuf>,
    // logs & results
    log: String,
    last_json_path: Option<PathBuf>,
    results: Vec<WheelOne>,
    // config
    python_bin: String,
    weights_path: String,
    project_root: String,
    // caches
    tex_cache: HashMap<String, egui::TextureHandle>,
    last_run_bbox_dir: Option<PathBuf>,
    // UI selection
    selected_image: Option<String>,
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            pending_files: vec![],
            log: String::new(),
            last_json_path: None,
            results: vec![],
            python_bin: "python3".to_string(),
            weights_path: "yolov8/train_result/ver14/weights/best.pt".to_string(),  // 학습한 모델중 가장 성능이 좋은 ver14 사용
            project_root: ".".to_string(),
            tex_cache: HashMap::new(),
            last_run_bbox_dir: None,
            selected_image: None,
        }
    }
}

impl eframe::App for AppState {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        ctx.set_debug_on_hover(false);

        // drag & drop
        for dropped in &ctx.input(|i| i.raw.dropped_files.clone()) {
            if let Some(path) = &dropped.path { self.pending_files.push(path.clone()); }
        }

        egui::TopBottomPanel::top("top").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("Wheel City AI 2 – Runner");
                if ui.button("Close").clicked() {
                    ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                }
            });
        });

        egui::SidePanel::left("left").resizable(true).show(ctx, |ui| {
            ui.group(|ui| {
                ui.label("Python executable (path or command)");
                ui.text_edit_singleline(&mut self.python_bin);
            });

            ui.add_space(8.0);
            ui.group(|ui| {
                ui.label("YOLO weights path (best.pt)");
                ui.text_edit_singleline(&mut self.weights_path);
                if ui.button("Select file").clicked() {
                    if let Some(p) = FileDialog::new().add_filter("pt", &["pt"]).pick_file() {
                        self.weights_path = p.to_string_lossy().to_string();
                    }
                }
            });

            ui.add_space(8.0);
            ui.group(|ui| {
                if ui.button("Select images...").clicked() {
                    if let Some(files) = FileDialog::new()
                        .add_filter("images", &["jpg","jpeg","png","webp","bmp"])
                        .pick_files()
                    {
                        self.pending_files.extend(files);
                    }
                }
                ui.add_space(4.0);
                ui.label("You can also drag & drop images here.");
                ui.separator();
                ui.label(egui::RichText::new("Pending images").strong());
                let mut remove_idx: Option<usize> = None;
                for (i, p) in self.pending_files.iter().enumerate() {
                    ui.horizontal(|ui| {
                        ui.label(format!("• {}", p.display()));
                        if ui.small_button("Remove").clicked() { remove_idx = Some(i); }
                    });
                }
                if let Some(i) = remove_idx { self.pending_files.remove(i); }
                ui.add_space(8.0);

                if ui.button(egui::RichText::new("▶ Run").color(Color32::WHITE)).clicked() {
                    if let Err(e) = self.run_pipeline() {
                        self.append_log(&format!("[ERROR] {}\n", e));
                    }
                    ctx.request_repaint();
                }
            });

            ui.add_space(12.0);
            ui.separator();
            ui.label(egui::RichText::new("Log").strong());
            egui::ScrollArea::vertical()
                .id_source("log_scroll")
                .max_height(220.0)
                .show(ui, |ui| { ui.monospace(&self.log); });

            ui.add_space(8.0);
            if let Some(p) = &self.last_json_path {
                ui.label(format!("Last result JSON: {}", p.display()));
            }
            if let Some(d) = &self.last_run_bbox_dir {
                ui.label(format!("Last run bbox dir: {}", d.display()));
            }
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            // ===== Results table =====
            ui.heading("Results preview");
            ui.add_space(6.0);

            let rows = self.results.clone(); // avoid borrow conflicts

            egui::ScrollArea::vertical()
                .id_source("results_scroll")
                .auto_shrink([false; 2])
                .show(ui, |ui| {
                    TableBuilder::new(ui)
                        .striped(true)
                        .column(Column::auto().at_least(78.0))    // BBox thumb
                        .column(Column::auto().at_least(200.0))   // Image name
                        .column(Column::auto().at_least(110.0))   // Accessible
                        .column(Column::remainder())               // Reason (ellipsized)
                        .header(22.0, |mut header| {
                            header.col(|ui| { ui.strong("BBox"); });
                            header.col(|ui| { ui.strong("Image"); });
                            header.col(|ui| { ui.strong("Accessible"); });
                            header.col(|ui| { ui.strong("Reason"); });
                        })
                        .body(|mut body| {
                            for r in rows {
                                let is_selected = self.selected_image.as_deref() == Some(r.image.as_str());
                                body.row(28.0, |mut row| {
                                    // thumb
                                    row.col(|ui| { self.show_bbox_thumb(ui, &r.image, ctx); });
                                    // filename (click to select)
                                    row.col(|ui| {
                                        let resp = ui.selectable_label(is_selected, &r.image);
                                        if resp.clicked() {
                                            self.selected_image = Some(r.image.clone());
                                        }
                                    });
                                    // accessible
                                    row.col(|ui| match r.result.accessible {
                                        Some(true)  => { ui.colored_label(Color32::from_rgb(0,160,0), "true"); }
                                        Some(false) => { ui.colored_label(Color32::from_rgb(200,0,0), "false"); }
                                        None        => { ui.label("null"); }
                                    });
                                    // reason (single line, ellipsized to avoid overlap)
                                    row.col(|ui| {
                                        ui.add(egui::Label::new(egui::RichText::new(&r.result.reason)).truncate(true).wrap(false));
                                    });
                                });
                            }
                        });
                });

            ui.separator();
            ui.add_space(6.0);

            // ===== Bottom area: Big bbox preview + full reason =====
            ui.horizontal(|ui| {
                ui.vertical(|ui| {
                    ui.heading("Selected BBox image");
                    ui.label("Click a filename in the table to select.");
                    ui.add_space(6.0);
                    egui::ScrollArea::both()
                        .id_source("big_preview_scroll")
                        .show(ui, |ui| {
                            if let Some(p) = self.find_bbox_image_for_selected() {
                                let key = format!("big:{}", p.display());
                                if !self.tex_cache.contains_key(&key) {
                                    if let Some(tex) = load_texture_from_path(ctx, &p) {
                                        self.tex_cache.insert(key.clone(), tex);
                                    }
                                }
                                if let Some(tex) = self.tex_cache.get(&key) {
                                    let size = tex.size_vec2();
                                    let max_w = ui.available_width().min(1400.0);
                                    let scale = (max_w / size.x).min(1.0);
                                    let sized = egui::load::SizedTexture::from_handle(tex);
                                    egui::Image::new(sized)
                                        .max_width(size.x * scale)
                                        .max_height(size.y * scale)
                                        .ui(ui);
                                } else {
                                    ui.label("Failed to load selected image.");
                                }
                            } else {
                                ui.label("Select a row above to preview its bbox image here.");
                            }
                        });
                });

                ui.separator();

                ui.vertical(|ui| {
                    ui.heading("Full reason");
                    ui.add_space(6.0);
                    egui::ScrollArea::vertical()
                        .id_source("reason_full_scroll")
                        .show(ui, |ui| {
                            if let Some(sel) = self.selected_image.as_ref() {
                                if let Some(item) = self.results.iter().find(|w| &w.image == sel) {
                                    // wrapped, full text
                                    ui.label(egui::RichText::new(&item.result.reason));
                                } else {
                                    ui.label("No reason available.");
                                }
                            } else {
                                ui.label("Select a row to view the full reason.");
                            }
                        });
                });
            });
        });
    }
}

impl AppState {
    fn append_log(&mut self, s: &str) {
        self.log.push_str(s);
        if self.log.len() > 120_000 { self.log = self.log[self.log.len() - 60_000..].to_string(); }
    }

    fn run_pipeline(&mut self) -> Result<()> {
        let project_root = self.resolve_project_root()?;
        if project_root.to_string_lossy() != self.project_root {
            self.append_log(&format!("[INFO] project root auto-detected: {}\n", project_root.display()));
        }

        let python = self.resolve_python(&project_root)?;
        self.append_log(&format!("[INFO] using Python: {}\n", python));

        let yolo_script   = project_root.join("yolov8").join("run.py");
        let gemini_script = project_root.join("gemini").join("run.py");
        if !yolo_script.exists()  { anyhow::bail!("Missing script: {}", yolo_script.display()); }
        if !gemini_script.exists(){ anyhow::bail!("Missing script: {}", gemini_script.display()); }
        let weights_abs = project_root.join(&self.weights_path);
        if !weights_abs.exists()  { anyhow::bail!("Weights file not found: {}", weights_abs.display()); }

        // user-visible
        let user_input_dir = project_root.join("input_images");
        fs::create_dir_all(&user_input_dir).ok();

        // run-scoped
        let work_dir   = project_root.join(".runner_work");
        let run_input  = work_dir.join("input");
        let ts         = Local::now().format("%Y%m%d_%H%M%S").to_string();
        let run_bbox   = work_dir.join("bbox").join(&ts);
        let results_dir= project_root.join("results");
        fs::create_dir_all(&run_input).ok();
        fs::create_dir_all(&run_bbox).ok();
        fs::create_dir_all(&results_dir).ok();

        // clear run_input only
        for e in fs::read_dir(&run_input)? {
            let p = e?.path();
            if p.is_file() { let _ = fs::remove_file(p); }
        }

        // sources
        let sources: Vec<PathBuf> = if self.pending_files.is_empty() {
            let mut v = vec![];
            if let Ok(rd) = fs::read_dir(&user_input_dir) {
                for e in rd.flatten() {
                    let p = e.path();
                    if p.is_file() { v.push(p); }
                }
            }
            v
        } else {
            self.pending_files.clone()
        };

        // copy into run_input with unique names
        self.append_log("[STEP] copying into work input dir...\n");
        let mut used_names: HashSet<String> = HashSet::new();
        for src in &sources {
            if !src.exists() {
                self.append_log(&format!("[WARN] source missing, skip: {}\n", src.display()));
                continue;
            }
            let base = src.file_name().unwrap().to_string_lossy().to_string();
            let mut final_name = base.clone();
            let mut counter = 1;
            while used_names.contains(&final_name) || run_input.join(&final_name).exists() {
                let (stem, ext) = split_name_ext(&base);
                final_name = format!("{}_{}{}", stem, counter, ext);
                counter += 1;
            }
            let dst = run_input.join(&final_name);
            if let Err(e) = fs::copy(src, &dst) {
                self.append_log(&format!("[WARN] copy failed (skip): {} -> {} ({})\n", src.display(), dst.display(), e));
            } else {
                used_names.insert(final_name);
            }
        }

        // YOLO → run_bbox
        self.append_log("[STEP] running YOLO inference...\n");
        let mut cmd = Command::new(&python);
        cmd.arg(&yolo_script)
           .arg("--weights").arg(&weights_abs)
           .arg("--source").arg(&run_input)
           .arg("--outdir").arg(&run_bbox);
        self.exec_and_log_in_dir(cmd, "[YOLO] ", &project_root)?;

        // Gemini
        self.append_log("[STEP] running Gemini judgment...\n");
        let out_json = results_dir.join(format!("result_{}.json", ts));
        let mut cmd2 = Command::new(&python);
        cmd2.arg(&gemini_script)
            .arg("--images_dir").arg(&run_bbox)
            .arg("--out_json").arg(&out_json);
        self.exec_and_log_in_dir(cmd2, "[GEMINI] ", &project_root)?;

        // load results
        self.append_log("[STEP] loading results...\n");
        let data = fs::read_to_string(&out_json).with_context(|| "failed to read result json")?;
        let parsed: WheelResultFile = serde_json::from_str(&data).with_context(|| "failed to parse result json")?;
        self.results = parsed.results;
        self.last_json_path = Some(out_json.clone());
        self.last_run_bbox_dir = Some(run_bbox.clone());

        // auto-select first item
        if self.selected_image.is_none() {
            if let Some(first) = self.results.first() {
                self.selected_image = Some(first.image.clone());
            }
        }

        // clear caches for new run
        self.tex_cache.clear();

        self.append_log("[DONE] Completed.\n");
        Ok(())
    }

    fn exec_and_log_in_dir(&mut self, mut cmd: Command, prefix: &str, workdir: &Path) -> Result<()> {
        cmd.current_dir(workdir);
        let out = cmd.output().with_context(|| "failed to spawn process")?;
        if !out.stdout.is_empty() { self.append_log(&format!("{}{}", prefix, String::from_utf8_lossy(&out.stdout))); }
        if !out.stderr.is_empty() { self.append_log(&format!("{}[stderr] {}", prefix, String::from_utf8_lossy(&out.stderr))); }
        if !out.status.success() { anyhow::bail!("subprocess failed with code {:?}", out.status.code()); }
        Ok(())
    }

    fn resolve_python(&mut self, project_root: &Path) -> Result<String> {
        let mut candidates: Vec<String> = vec![
            project_root.join(".venv").join("bin").join("python").to_string_lossy().to_string(),
            project_root.join(".venv").join("Scripts").join("python.exe").to_string_lossy().to_string(),
        ];
        if !self.python_bin.trim().is_empty() { candidates.push(self.python_bin.clone()); }
        candidates.push("python3".to_string());
        candidates.push("python".to_string());

        for cand in candidates {
            if Command::new(&cand).arg("--version").output().is_ok() { return Ok(cand); }
        }
        Err(anyhow::anyhow!(
            "No working Python found. Create venv at {}/.venv or set an explicit path.",
            project_root.display()
        ))
    }

    fn resolve_project_root(&mut self) -> Result<PathBuf> {
        let mut cands: Vec<PathBuf> = vec![PathBuf::from(self.project_root.clone())];
        let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        cands.push(cwd.clone());
        for p in cwd.ancestors().skip(1).take(5) { cands.push(p.to_path_buf()); }
        cands.sort(); cands.dedup();
        for cand in cands {
            if looks_like_repo_root(&cand) {
                self.project_root = cand.to_string_lossy().to_string();
                return Ok(PathBuf::from(&self.project_root));
            }
        }
        Err(anyhow::anyhow!("Could not locate project root containing yolov8/run.py and gemini/run.py"))
    }

    // small thumb in table
    fn show_bbox_thumb(&mut self, ui: &mut egui::Ui, filename: &str, ctx: &egui::Context) {
        if let Some(p) = self.find_bbox_image_path(filename) {
            let key = format!("thumb:{}", p.display());
            if !self.tex_cache.contains_key(&key) {
                if let Some(tex) = load_texture_from_path(ctx, &p) {
                    self.tex_cache.insert(key.clone(), tex);
                }
            }
            if let Some(tex) = self.tex_cache.get(&key) {
                let sized = egui::load::SizedTexture::from_handle(tex);
                egui::Image::new(sized).max_width(72.0).max_height(54.0).ui(ui);
                return;
            }
        }
        ui.label("—");
    }

    // resolve selected image full path (robust to extension mismatches)
    fn find_bbox_image_for_selected(&self) -> Option<PathBuf> {
        if let Some(sel) = self.selected_image.as_ref() {
            self.find_bbox_image_path(sel)
        } else { None }
    }

    fn find_bbox_image_path(&self, filename: &str) -> Option<PathBuf> {
        let dir = self.last_run_bbox_dir.as_ref()?;
        let direct = dir.join(filename);
        if direct.exists() { return Some(direct); }
        // fallback: search by stem across extensions
        let stem = Path::new(filename).file_stem()?.to_string_lossy().to_string();
        let exts = ["jpg","jpeg","png","webp","bmp"];
        for e in &exts {
            let cand = dir.join(format!("{}.{}", stem, e));
            if cand.exists() { return Some(cand); }
        }
        // as a last resort, scan all files in dir and match stem
        if let Ok(rd) = fs::read_dir(dir) {
            for ent in rd.flatten() {
                let p = ent.path();
                if p.is_file() {
                    if let Some(s) = p.file_stem().map(|s| s.to_string_lossy().to_string()) {
                        if s == stem { return Some(p); }
                    }
                }
            }
        }
        None
    }
}

fn looks_like_repo_root(dir: &Path) -> bool {
    dir.join("yolov8").join("run.py").exists() &&
    dir.join("gemini").join("run.py").exists()
}

fn split_name_ext(name: &str) -> (String, String) {
    let p = Path::new(name);
    let stem = p.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_else(|| name.to_string());
    let ext  = p.extension().map(|e| format!(".{}", e.to_string_lossy())).unwrap_or_default();
    (stem, ext)
}

fn load_texture_from_path(ctx: &egui::Context, path: &Path) -> Option<egui::TextureHandle> {
    let data = fs::read(path).ok()?;
    let img = image::load_from_memory(&data).ok()?;
    let rgba = img.to_rgba8();
    let size = [img.width() as usize, img.height() as usize];
    let color_img = egui::ColorImage::from_rgba_unmultiplied(size, rgba.as_raw());
    Some(ctx.load_texture(
        path.file_name()?.to_string_lossy(),
        color_img,
        egui::TextureOptions::default(),
    ))
}

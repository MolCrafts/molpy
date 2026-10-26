//! LAMMPS log file parser.

use serde::Serialize;

use crate::io::Source;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Header text before the first parsed LAMMPS run block.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct LammpsLogHeader {
    pub lines: Vec<String>,
}

impl LammpsLogHeader {
    /// Header lines joined by newlines.
    pub fn raw_text(&self) -> String {
        self.lines.join("\n")
    }
}

/// ``Per MPI rank memory allocation`` line.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct LammpsMemoryUsage {
    pub minimum: f64,
    pub average: f64,
    pub maximum: f64,
    pub units: String,
    pub raw_line: String,
}

/// LAMMPS thermo table with dynamic columns.
///
/// Rows are plain `Vec<f64>` (column-major names in [`Self::columns`]) so the
/// result serializes cleanly to JSON / Python without a structured-array dtype.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct LammpsThermo {
    pub columns: Vec<String>,
    pub rows: Vec<Vec<f64>>,
    pub raw_lines: Vec<String>,
}

impl LammpsThermo {
    /// Number of thermo rows.
    pub fn n_rows(&self) -> usize {
        self.rows.len()
    }
}

/// ``Loop time`` summary line.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct LammpsLoopTime {
    pub seconds: f64,
    pub procs: i64,
    pub steps: Option<i64>,
    pub atoms: Option<i64>,
    pub raw_line: String,
}

/// LAMMPS ``Performance`` summary line.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct LammpsPerformance {
    pub ns_per_day: f64,
    pub hours_per_ns: f64,
    pub timesteps_per_second: f64,
    pub atom_steps_per_second: Option<f64>,
    pub atom_steps_units: Option<String>,
    pub raw_line: String,
}

/// ``% CPU use`` summary line.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct LammpsCpuUse {
    pub percent: f64,
    #[serde(rename = "MPI_tasks")]
    pub mpi_tasks: i64,
    #[serde(rename = "OMP_threads")]
    pub omp_threads: Option<i64>,
    pub raw_line: String,
}

/// One row from a LAMMPS timing breakdown table.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct LammpsTimingRow {
    pub section: String,
    pub min_time: f64,
    pub avg_time: f64,
    pub max_time: f64,
    pub percent_varavg: f64,
    pub percent_total: f64,
    pub raw_line: String,
}

/// ``MPI task timing breakdown`` or thread timing table.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct LammpsTimingBreakdown {
    pub title: String,
    pub rows: Vec<LammpsTimingRow>,
    pub raw_lines: Vec<String>,
}

/// LAMMPS load-balance statistic plus optional histogram.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct LammpsLoadBalance {
    pub name: String,
    pub average: f64,
    pub maximum: f64,
    pub minimum: f64,
    pub histogram: Vec<i64>,
    pub raw_lines: Vec<String>,
}

/// Neighbor-list statistics emitted after a run.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct LammpsNeighborStatistics {
    pub total_neighbors: Option<i64>,
    pub ave_neighs_per_atom: Option<f64>,
    pub ave_special_neighs_per_atom: Option<f64>,
    pub neighbor_list_builds: Option<i64>,
    pub dangerous_builds: Option<i64>,
    pub raw_lines: Vec<String>,
}

/// A warning line from the LAMMPS log.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct LammpsWarning {
    pub message: String,
    pub raw_line: String,
    pub line_number: Option<usize>,
    pub run_index: Option<usize>,
}

/// One LAMMPS run output block.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct LammpsRun {
    pub index: usize,
    pub setup_log: Vec<String>,
    pub memory: Option<LammpsMemoryUsage>,
    pub thermo: Option<LammpsThermo>,
    pub loop_time: Option<LammpsLoopTime>,
    pub performance: Option<LammpsPerformance>,
    #[serde(rename = "CPU_use")]
    pub cpu_use: Option<LammpsCpuUse>,
    #[serde(rename = "MPI_task_timing")]
    pub mpi_task_timing: Option<LammpsTimingBreakdown>,
    pub thread_timing: Option<LammpsTimingBreakdown>,
    pub load_balance: Vec<LammpsLoadBalance>,
    pub neighbor_statistics: Option<LammpsNeighborStatistics>,
    pub warnings: Vec<LammpsWarning>,
    pub unparsed_log: Vec<String>,
    pub raw_text: String,
}

/// Parsed LAMMPS log with one structured entry per run.
///
/// The parser follows the standard LAMMPS run output documented in
/// `Run_output.html`: thermo tables, loop timing, performance summaries,
/// CPU/MPI timing, load-balance statistics, neighbor statistics, and warnings.
/// Unrecognized lines are preserved so callers can still inspect information
/// that does not yet have a structured representation.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct LammpsLog {
    pub path: String,
    pub version: Option<String>,
    pub header: LammpsLogHeader,
    pub runs: Vec<LammpsRun>,
    pub total_wall_time: Option<String>,
    pub warnings: Vec<LammpsWarning>,
    pub raw_text: String,
    pub style: String,
}

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

/// Read a LAMMPS log.
///
/// `style` is the thermo style the run printed; only `"default"` thermo
/// tables are currently parsed. The result's [`path`](LammpsLog::path) is the
/// file read, and empty for bytes.
///
/// # Errors
///
/// Returns [`std::io::Error`] if the file cannot be read, or `InvalidData`
/// for a log that is not UTF-8 text.
pub fn read_lammps_log<'a>(src: impl Into<Source<'a>>, style: &str) -> std::io::Result<LammpsLog> {
    let src = src.into();
    let name = src
        .path()
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_default();
    Ok(parse_log(&src.text()?, &name, style))
}

/// Parse a LAMMPS log's text; `source_name` is recorded as the result's
/// [`path`](LammpsLog::path) and never opened.
fn parse_log(text: &str, source_name: &str, style: &str) -> LammpsLog {
    let lines: Vec<&str> = text.lines().collect();
    let run_ranges = find_run_ranges(&lines);
    let mut header_end = run_ranges.first().map(|(s, _)| *s).unwrap_or(lines.len());
    let total_wall_time = parse_total_wall_time(&lines);
    if total_wall_time.is_some() && run_ranges.is_empty() {
        let wall_idx = total_wall_time_index(&lines);
        if wall_idx >= 0 {
            header_end = header_end.min(wall_idx as usize);
        }
    }

    let version = first_line(text);
    let header = LammpsLogHeader {
        lines: lines[..header_end]
            .iter()
            .map(|s| (*s).to_string())
            .collect(),
    };
    let runs = run_ranges
        .iter()
        .enumerate()
        .map(|(index, &(start, end))| parse_run(index, &lines[start..end], start, style))
        .collect();
    let warnings = collect_warnings(&lines, 0, None);

    LammpsLog {
        path: source_name.to_string(),
        version,
        header,
        runs,
        total_wall_time,
        warnings,
        raw_text: text.to_string(),
        style: style.to_string(),
    }
}

/// Whether `text` holds a LAMMPS run: a `Per MPI rank memory allocation
/// (min/avg/max) = …` line, the line [`read_lammps_log`] opens each run
/// at.
///
/// The banner alone is not enough — a log whose run died during setup has
/// no run to read — so this asks for what the reader keys on. Cheap enough to
/// run on a file's first bytes before deciding to read the whole thing.
pub fn is_lammps_log(text: &str) -> bool {
    text.lines().any(|line| match_memory(line.trim()).is_some())
}

// ---------------------------------------------------------------------------
// Parsing
// ---------------------------------------------------------------------------

fn find_run_ranges(lines: &[&str]) -> Vec<(usize, usize)> {
    let starts: Vec<usize> = lines
        .iter()
        .enumerate()
        .filter_map(|(idx, line)| {
            if match_memory(line.trim()).is_some() {
                Some(idx)
            } else {
                None
            }
        })
        .collect();
    // A log written with `log ... append` holds one `Total wall time:` per
    // LAMMPS invocation, not one per file. Clamping every run to the *first*
    // of them put the end before the start for any run belonging to a later
    // invocation, and `lines[start..end]` then panicked. Each run ends at the
    // first wall-time line that follows it.
    let wall_idxs = total_wall_time_indices(lines);
    let mut ranges = Vec::with_capacity(starts.len());
    for (i, &start) in starts.iter().enumerate() {
        let next_start = starts.get(i + 1).copied().unwrap_or(lines.len());
        let wall_index = wall_idxs.partition_point(|&wall| wall <= start);
        let end = match wall_idxs.get(wall_index) {
            Some(&wall) => next_start.min(wall),
            None => next_start,
        };
        ranges.push((start, end.max(start)));
    }
    ranges
}

fn parse_run(index: usize, lines: &[&str], line_offset: usize, style: &str) -> LammpsRun {
    let mut consumed = vec![false; lines.len()];

    let memory = parse_memory(lines, &mut consumed);
    let loop_idx = first_index(lines, |line| match_loop_time(line.trim()).is_some());
    let thermo = parse_thermo(lines, &mut consumed, loop_idx, style);
    let loop_time = parse_loop_time(lines, &mut consumed);
    let performance = parse_performance(lines, &mut consumed);
    let cpu_use = parse_cpu_use(lines, &mut consumed);
    let mpi_task_timing = parse_timing_breakdown(lines, &mut consumed, "MPI task timing breakdown");
    let mut thread_timing =
        parse_timing_breakdown(lines, &mut consumed, "Thread timings breakdown");
    if thread_timing.is_none() {
        thread_timing = parse_timing_breakdown(lines, &mut consumed, "Thread timing");
    }

    let load_balance = parse_load_balance(lines, &mut consumed);
    let neighbor_statistics = parse_neighbor_statistics(lines, &mut consumed);
    let warnings = collect_warnings(lines, line_offset, Some(index));
    for (idx, line) in lines.iter().enumerate() {
        if match_warning(line.trim()).is_some() {
            consumed[idx] = true;
        }
    }

    let setup_end = if loop_idx >= 0 { loop_idx as usize } else { 0 };
    let setup_log: Vec<String> = lines[..setup_end]
        .iter()
        .enumerate()
        .filter(|(idx, line)| !consumed[*idx] && !line.trim().is_empty())
        .map(|(_, line)| (*line).to_string())
        .collect();
    let setup_set: std::collections::HashSet<&str> = setup_log.iter().map(String::as_str).collect();
    let unparsed_log: Vec<String> = lines
        .iter()
        .enumerate()
        .filter(|(idx, line)| {
            !consumed[*idx] && !line.trim().is_empty() && !setup_set.contains(*line)
        })
        .map(|(_, line)| (*line).to_string())
        .collect();

    LammpsRun {
        index,
        setup_log,
        memory,
        thermo,
        loop_time,
        performance,
        cpu_use,
        mpi_task_timing,
        thread_timing,
        load_balance,
        neighbor_statistics,
        warnings,
        unparsed_log,
        raw_text: lines.join("\n"),
    }
}

fn parse_memory(lines: &[&str], consumed: &mut [bool]) -> Option<LammpsMemoryUsage> {
    for (idx, line) in lines.iter().enumerate() {
        if let Some(mem) = match_memory(line.trim()) {
            consumed[idx] = true;
            return Some(LammpsMemoryUsage {
                minimum: mem.0,
                average: mem.1,
                maximum: mem.2,
                units: mem.3,
                raw_line: (*line).to_string(),
            });
        }
    }
    None
}

fn parse_thermo(
    lines: &[&str],
    consumed: &mut [bool],
    loop_idx: i64,
    style: &str,
) -> Option<LammpsThermo> {
    if style != "default" || loop_idx < 0 {
        return None;
    }

    let mut start = first_index(lines, |line| match_memory(line.trim()).is_some());
    if start < 0 {
        start = -1;
    }
    let body_indices: Vec<usize> = ((start + 1) as usize..loop_idx as usize)
        .filter(|&idx| !lines[idx].trim().is_empty())
        .collect();
    if body_indices.len() < 2 {
        return None;
    }

    let header_idx = body_indices[0];
    let columns: Vec<String> = lines[header_idx]
        .split_whitespace()
        .map(str::to_string)
        .collect();
    if columns.is_empty() {
        return None;
    }

    let mut data_indices: Vec<usize> = Vec::new();
    for &idx in &body_indices[1..] {
        let parts: Vec<&str> = lines[idx].split_whitespace().collect();
        if parts.len() != columns.len() || !parts.iter().all(|p| is_float(p)) {
            break;
        }
        data_indices.push(idx);
    }
    if data_indices.is_empty() {
        return None;
    }

    let mut rows = Vec::with_capacity(data_indices.len());
    for &idx in &data_indices {
        let row: Result<Vec<f64>, _> = lines[idx]
            .split_whitespace()
            .map(|p| p.parse::<f64>())
            .collect();
        match row {
            Ok(r) => rows.push(r),
            Err(_) => return None,
        }
    }

    consumed[header_idx] = true;
    for &idx in &data_indices {
        consumed[idx] = true;
    }
    let mut raw_lines = Vec::with_capacity(1 + data_indices.len());
    raw_lines.push(lines[header_idx].to_string());
    for &idx in &data_indices {
        raw_lines.push(lines[idx].to_string());
    }

    Some(LammpsThermo {
        columns,
        rows,
        raw_lines,
    })
}

fn parse_loop_time(lines: &[&str], consumed: &mut [bool]) -> Option<LammpsLoopTime> {
    for (idx, line) in lines.iter().enumerate() {
        if let Some(lt) = match_loop_time(line.trim()) {
            consumed[idx] = true;
            return Some(LammpsLoopTime {
                seconds: lt.0,
                procs: lt.1,
                steps: lt.2,
                atoms: lt.3,
                raw_line: (*line).to_string(),
            });
        }
    }
    None
}

fn parse_performance(lines: &[&str], consumed: &mut [bool]) -> Option<LammpsPerformance> {
    for (idx, line) in lines.iter().enumerate() {
        if let Some(perf) = match_performance(line.trim()) {
            consumed[idx] = true;
            return Some(LammpsPerformance {
                ns_per_day: perf.0,
                hours_per_ns: perf.1,
                timesteps_per_second: perf.2,
                atom_steps_per_second: perf.3,
                atom_steps_units: perf.4,
                raw_line: (*line).to_string(),
            });
        }
    }
    None
}

fn parse_cpu_use(lines: &[&str], consumed: &mut [bool]) -> Option<LammpsCpuUse> {
    for (idx, line) in lines.iter().enumerate() {
        if let Some(cpu) = match_cpu_use(line.trim()) {
            consumed[idx] = true;
            return Some(LammpsCpuUse {
                percent: cpu.0,
                mpi_tasks: cpu.1,
                omp_threads: cpu.2,
                raw_line: (*line).to_string(),
            });
        }
    }
    None
}

fn parse_timing_breakdown(
    lines: &[&str],
    consumed: &mut [bool],
    title_prefix: &str,
) -> Option<LammpsTimingBreakdown> {
    let start = first_index(lines, |line| line.trim().starts_with(title_prefix));
    if start < 0 {
        return None;
    }
    let start = start as usize;

    let mut raw_indices = vec![start];
    let mut rows = Vec::new();
    for (idx, line) in lines.iter().enumerate().skip(start + 1) {
        let stripped = line.trim();
        if stripped.is_empty() {
            break;
        }
        if let Some(row) = parse_timing_row(line) {
            rows.push(row);
            raw_indices.push(idx);
            continue;
        }
        if line.contains('|')
            || stripped.chars().all(|c| c == '-')
            || stripped.starts_with("Section")
        {
            raw_indices.push(idx);
            continue;
        }
        if !rows.is_empty() {
            break;
        }
    }

    if rows.is_empty() {
        return None;
    }
    for &idx in &raw_indices {
        consumed[idx] = true;
    }
    let title = lines[start].trim().trim_end_matches(':').to_string();
    Some(LammpsTimingBreakdown {
        title,
        rows,
        raw_lines: raw_indices.iter().map(|&i| lines[i].to_string()).collect(),
    })
}

fn parse_timing_row(line: &str) -> Option<LammpsTimingRow> {
    if !line.contains('|') {
        return None;
    }
    let parts: Vec<&str> = line.split('|').map(str::trim).collect();
    if parts.len() != 6 || !parts[1..].iter().all(|p| is_float(p)) {
        return None;
    }
    Some(LammpsTimingRow {
        section: parts[0].to_string(),
        min_time: parts[1].parse().ok()?,
        avg_time: parts[2].parse().ok()?,
        max_time: parts[3].parse().ok()?,
        percent_varavg: parts[4].parse().ok()?,
        percent_total: parts[5].parse().ok()?,
        raw_line: line.to_string(),
    })
}

fn parse_load_balance(lines: &[&str], consumed: &mut [bool]) -> Vec<LammpsLoadBalance> {
    let mut entries = Vec::new();
    let mut idx = 0;
    while idx < lines.len() {
        let Some((name, average, maximum, minimum)) = match_load_balance(lines[idx].trim()) else {
            idx += 1;
            continue;
        };

        let mut raw_indices = vec![idx];
        let mut histogram = Vec::new();
        let next_idx = idx + 1;
        if next_idx < lines.len() && lines[next_idx].trim().starts_with("Histogram:") {
            histogram = lines[next_idx]
                .split_whitespace()
                .skip(1)
                .filter_map(|v| v.parse::<i64>().ok())
                .collect();
            raw_indices.push(next_idx);
        }
        for &i in &raw_indices {
            consumed[i] = true;
        }
        entries.push(LammpsLoadBalance {
            name,
            average,
            maximum,
            minimum,
            histogram,
            raw_lines: raw_indices.iter().map(|&i| lines[i].to_string()).collect(),
        });
        idx = raw_indices[raw_indices.len() - 1] + 1;
    }
    entries
}

fn parse_neighbor_statistics(
    lines: &[&str],
    consumed: &mut [bool],
) -> Option<LammpsNeighborStatistics> {
    let mut total_neighbors = None;
    let mut ave_neighs_per_atom = None;
    let mut ave_special_neighs_per_atom = None;
    let mut neighbor_list_builds = None;
    let mut dangerous_builds = None;
    let mut raw_indices = Vec::new();

    for (idx, line) in lines.iter().enumerate() {
        if !line.contains('=') {
            continue;
        }
        let mut parts = line.splitn(2, '=');
        let key = parts.next()?.trim();
        let raw_value = parts.next()?.trim();
        match key {
            "Total # of neighbors" => {
                total_neighbors = parse_int_from_float_str(raw_value);
                raw_indices.push(idx);
            }
            "Ave neighs/atom" => {
                ave_neighs_per_atom = raw_value.parse().ok();
                raw_indices.push(idx);
            }
            "Ave special neighs/atom" => {
                ave_special_neighs_per_atom = raw_value.parse().ok();
                raw_indices.push(idx);
            }
            "Neighbor list builds" => {
                neighbor_list_builds = parse_int_from_float_str(raw_value);
                raw_indices.push(idx);
            }
            "Dangerous builds" => {
                dangerous_builds = parse_int_from_float_str(raw_value);
                raw_indices.push(idx);
            }
            _ => {}
        }
    }

    if raw_indices.is_empty() {
        return None;
    }
    for &idx in &raw_indices {
        consumed[idx] = true;
    }
    Some(LammpsNeighborStatistics {
        total_neighbors,
        ave_neighs_per_atom,
        ave_special_neighs_per_atom,
        neighbor_list_builds,
        dangerous_builds,
        raw_lines: raw_indices.iter().map(|&i| lines[i].to_string()).collect(),
    })
}

fn collect_warnings(
    lines: &[&str],
    line_offset: usize,
    run_index: Option<usize>,
) -> Vec<LammpsWarning> {
    let mut warnings = Vec::new();
    for (idx, line) in lines.iter().enumerate() {
        if let Some(message) = match_warning(line.trim()) {
            warnings.push(LammpsWarning {
                message,
                raw_line: (*line).to_string(),
                line_number: Some(line_offset + idx + 1),
                run_index,
            });
        }
    }
    warnings
}

fn parse_total_wall_time(lines: &[&str]) -> Option<String> {
    let idx = total_wall_time_index(lines);
    if idx < 0 {
        return None;
    }
    let line = lines[idx as usize];
    if let Some((_, rest)) = line.split_once(':') {
        Some(rest.trim().to_string())
    } else {
        Some(line.to_string())
    }
}

fn total_wall_time_index(lines: &[&str]) -> i64 {
    first_index(lines, |line| line.trim().starts_with("Total wall time:"))
}

/// Every `Total wall time:` line, ascending — one per LAMMPS invocation, so an
/// appended restart log has more than one.
fn total_wall_time_indices(lines: &[&str]) -> Vec<usize> {
    lines
        .iter()
        .enumerate()
        .filter(|(_, line)| line.trim().starts_with("Total wall time:"))
        .map(|(idx, _)| idx)
        .collect()
}

// ---------------------------------------------------------------------------
// Line matchers (no regex dependency)
// ---------------------------------------------------------------------------

/// Memory: `Per MPI rank memory allocation (min/avg/max) = A | B | C units`
fn match_memory(line: &str) -> Option<(f64, f64, f64, String)> {
    const PREFIX: &str = "Per MPI rank memory allocation (min/avg/max) = ";
    let rest = line.strip_prefix(PREFIX)?;
    let mut parts = rest.split('|');
    let minimum = parts.next()?.trim().parse().ok()?;
    let average = parts.next()?.trim().parse().ok()?;
    let tail = parts.next()?.trim();
    let mut tokens = tail.split_whitespace();
    let maximum = tokens.next()?.parse().ok()?;
    let units = tokens.next()?.to_string();
    if parts.next().is_some() {
        return None;
    }
    Some((minimum, average, maximum, units))
}

/// Loop: `Loop time of S on P procs` optional ` for N steps with A atoms`
fn match_loop_time(line: &str) -> Option<(f64, i64, Option<i64>, Option<i64>)> {
    const PREFIX: &str = "Loop time of ";
    let rest = line.strip_prefix(PREFIX)?;
    let (seconds_s, rest) = rest.split_once(" on ")?;
    let seconds = seconds_s.trim().parse().ok()?;
    let (procs_s, rest) = rest.split_once(" procs")?;
    let procs = procs_s.trim().parse().ok()?;
    let rest = rest.trim();
    if rest.is_empty() {
        return Some((seconds, procs, None, None));
    }
    // " for N steps with A atoms"
    let rest = rest.strip_prefix("for ")?.trim();
    let (steps_s, rest) = rest.split_once(" steps with ")?;
    let steps = steps_s.trim().parse().ok()?;
    let atoms_s = rest.strip_suffix(" atoms")?.trim();
    let atoms = atoms_s.parse().ok()?;
    Some((seconds, procs, Some(steps), Some(atoms)))
}

/// Performance: `Performance: A ns/day, B hours/ns, C timesteps/s` optional `, D units`
type PerformanceMatch = (f64, f64, f64, Option<f64>, Option<String>);
fn match_performance(line: &str) -> Option<PerformanceMatch> {
    const PREFIX: &str = "Performance:";
    let rest = line.strip_prefix(PREFIX)?.trim();
    let (ns_s, rest) = rest.split_once(" ns/day,")?;
    let ns_per_day = ns_s.trim().parse().ok()?;
    let (hours_s, rest) = rest.split_once(" hours/ns,")?;
    let hours_per_ns = hours_s.trim().parse().ok()?;
    let rest = rest.trim();
    // either "C timesteps/s" or "C timesteps/s, D units"
    let (tps_s, atom_part) = if let Some((a, b)) = rest.split_once(" timesteps/s,") {
        (a, Some(b.trim()))
    } else {
        let a = rest.strip_suffix(" timesteps/s")?;
        (a, None)
    };
    let timesteps_per_second = tps_s.trim().parse().ok()?;
    let (atom_steps, atom_units) = match atom_part {
        Some(tail) => {
            let mut tokens = tail.split_whitespace();
            let val = tokens.next()?.parse().ok()?;
            let units = tokens.next()?.to_string();
            (Some(val), Some(units))
        }
        None => (None, None),
    };
    Some((
        ns_per_day,
        hours_per_ns,
        timesteps_per_second,
        atom_steps,
        atom_units,
    ))
}

/// CPU: `P% CPU use with N MPI tasks` optional ` x T OpenMP threads`
fn match_cpu_use(line: &str) -> Option<(f64, i64, Option<i64>)> {
    let (percent_s, rest) = line.split_once("% CPU use with ")?;
    let percent = percent_s.trim().parse().ok()?;
    let (tasks_s, rest) = rest.split_once(" MPI tasks")?;
    let mpi_tasks = tasks_s.trim().parse().ok()?;
    let rest = rest.trim();
    if rest.is_empty() {
        return Some((percent, mpi_tasks, None));
    }
    let rest = rest.strip_prefix('x')?.trim();
    let threads_s = rest.strip_suffix(" OpenMP threads")?.trim();
    let omp = threads_s.parse().ok()?;
    Some((percent, mpi_tasks, Some(omp)))
}

/// Load balance: `Name: A ave B max C min`
fn match_load_balance(line: &str) -> Option<(String, f64, f64, f64)> {
    let (name, rest) = line.split_once(':')?;
    let name = name.trim();
    if !matches!(name, "Nlocal" | "Nghost" | "Neighs") {
        return None;
    }
    let rest = rest.trim();
    let (ave_s, rest) = rest.split_once(" ave ")?;
    let average = ave_s.trim().parse().ok()?;
    let (max_s, rest) = rest.split_once(" max ")?;
    let maximum = max_s.trim().parse().ok()?;
    let min_s = rest.strip_suffix(" min")?.trim();
    let minimum = min_s.parse().ok()?;
    Some((name.to_string(), average, maximum, minimum))
}

fn match_warning(line: &str) -> Option<String> {
    line.strip_prefix("WARNING:")
        .map(|m| m.trim_start().to_string())
}

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

fn first_index(lines: &[&str], predicate: impl Fn(&str) -> bool) -> i64 {
    for (idx, line) in lines.iter().enumerate() {
        if predicate(line) {
            return idx as i64;
        }
    }
    -1
}

fn first_line(text: &str) -> Option<String> {
    text.lines().next().map(str::to_string)
}

fn is_float(value: &str) -> bool {
    value.parse::<f64>().is_ok()
}

fn parse_int_from_float_str(value: &str) -> Option<i64> {
    // Python does converter(float(raw_value)) for int fields.
    value.parse::<f64>().ok().map(|v| v as i64)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_log_is_recognised_by_its_run_and_not_by_its_banner() {
        assert!(is_lammps_log(sample_full_log()));
        // A run that died in setup printed the banner but never a run.
        assert!(!is_lammps_log(
            "LAMMPS (1 Jan 2026)\nERROR: Unknown command\n"
        ));
        assert!(!is_lammps_log(""));
        // The reader agrees: what the sniffer refuses has no run.
        let banner_only = "LAMMPS (1 Jan 2026)\nTotal wall time: 0:00:00\n";
        assert!(!is_lammps_log(banner_only));
        assert!(
            read_lammps_log(banner_only.as_bytes(), "default")
                .unwrap()
                .runs
                .is_empty()
        );
    }

    fn sample_full_log() -> &'static str {
        r#"LAMMPS (1 Jan 2026)
using 1 OpenMP thread(s) per MPI task
Per MPI rank memory allocation (min/avg/max) = 4.5 | 4.75 | 5.0 Mbytes
Step Temp PotEng E_pair
0 300.0 -1000.0 -900.0
10 305.0 -1010.0 -910.0
Loop time of 0.2 on 2 procs for 10 steps with 100 atoms
Performance: 432.0 ns/day, 0.056 hours/ns, 50.0 timesteps/s, 5.0 katom-step/s
98.5% CPU use with 2 MPI tasks x 1 OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0.010 | 0.011 | 0.012 | 0.0 | 55.0
Neigh   | 0.002 | 0.003 | 0.004 | 0.0 | 15.0

Thread timings breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0.005 | 0.006 | 0.007 | 0.0 | 30.0

Nlocal:    50 ave 55 max 45 min
Histogram: 1 0 0 0 0 0 0 0 0 1
Nghost:    10 ave 12 max 8 min
Histogram: 0 1 0 0 0 0 0 0 1 0
Neighs:    150 ave 160 max 140 min
Histogram: 0 0 1 0 0 0 0 1 0 0
Total # of neighbors = 300
Ave neighs/atom = 3.0
Ave special neighs/atom = 2.0
Neighbor list builds = 1
Dangerous builds = 0
WARNING: test warning
Total wall time: 0:00:01
"#
    }

    #[test]
    fn parses_nested_run_structure() {
        let log = read_lammps_log(sample_full_log().as_bytes(), "default").unwrap();
        assert_eq!(log.version.as_deref(), Some("LAMMPS (1 Jan 2026)"));
        assert_eq!(log.total_wall_time.as_deref(), Some("0:00:01"));
        assert_eq!(log.runs.len(), 1);

        let run = &log.runs[0];
        assert_eq!(run.memory.as_ref().unwrap().average, 4.75);
        assert_eq!(
            run.thermo.as_ref().unwrap().columns,
            vec!["Step", "Temp", "PotEng", "E_pair"]
        );
        assert_eq!(run.thermo.as_ref().unwrap().rows[0][0], 0.0);
        assert_eq!(run.thermo.as_ref().unwrap().rows[1][0], 10.0);
        assert_eq!(run.loop_time.as_ref().unwrap().procs, 2);
        assert_eq!(run.loop_time.as_ref().unwrap().steps, Some(10));
        assert_eq!(run.loop_time.as_ref().unwrap().atoms, Some(100));
        assert_eq!(
            run.performance
                .as_ref()
                .unwrap()
                .atom_steps_units
                .as_deref(),
            Some("katom-step/s")
        );
        assert_eq!(run.cpu_use.as_ref().unwrap().mpi_tasks, 2);
        assert_eq!(run.cpu_use.as_ref().unwrap().omp_threads, Some(1));
        assert_eq!(
            run.mpi_task_timing.as_ref().unwrap().rows[0].section,
            "Pair"
        );
        assert_eq!(
            run.thread_timing.as_ref().unwrap().rows[0].percent_total,
            30.0
        );
        assert_eq!(run.load_balance[0].name, "Nlocal");
        assert_eq!(
            run.load_balance[0].histogram,
            vec![1, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        );
        assert_eq!(
            run.neighbor_statistics.as_ref().unwrap().total_neighbors,
            Some(300)
        );
        assert_eq!(
            run.neighbor_statistics.as_ref().unwrap().dangerous_builds,
            Some(0)
        );
        assert_eq!(run.warnings[0].message, "test warning");
        assert!(run.raw_text.contains("Step Temp PotEng E_pair"));
    }

    #[test]
    fn parses_two_stages() {
        let text = "\
LAMMPS (1 Jan 2026)
...
Per MPI rank memory allocation (min/avg/max) = 1 | 1 | 1 Mbytes
Step Temp PotEng
0 300.0 -1000.0
10 305.0 -1010.0
Loop time of 0.1 on 1 procs
...
Per MPI rank memory allocation (min/avg/max) = 1 | 1 | 1 Mbytes
Step Temp PotEng
20 310.0 -1020.0
30 315.0 -1030.0
Loop time of 0.1 on 1 procs
";
        let log = read_lammps_log(text.as_bytes(), "default").unwrap();
        assert_eq!(log.runs.len(), 2);
        assert_eq!(log.runs[0].thermo.as_ref().unwrap().rows[0][0], 0.0);
        assert_eq!(log.runs[1].thermo.as_ref().unwrap().rows[0][0], 20.0);
    }

    #[test]
    fn handles_no_thermo_block() {
        let text = "LAMMPS (1 Jan 2026)\n# nothing happened\n";
        let log = read_lammps_log(text.as_bytes(), "default").unwrap();
        assert!(log.runs.is_empty());
        assert!(log.version.as_ref().unwrap().starts_with("LAMMPS"));
    }

    #[test]
    fn matchers_cover_optional_tails() {
        assert!(match_loop_time("Loop time of 0.1 on 1 procs").is_some());
        assert_eq!(
            match_loop_time("Loop time of 0.2 on 2 procs for 10 steps with 100 atoms")
                .unwrap()
                .2,
            Some(10)
        );
        let perf = match_performance(
            "Performance: 432.0 ns/day, 0.056 hours/ns, 50.0 timesteps/s, 5.0 katom-step/s",
        )
        .unwrap();
        assert_eq!(perf.4.as_deref(), Some("katom-step/s"));
        let perf2 =
            match_performance("Performance: 0.071 ns/day, 336.080 hours/ns, 1.653 timesteps/s")
                .unwrap();
        assert!(perf2.3.is_none());
    }

    /// A log written with `log ... append` carries one `Total wall time:` per
    /// LAMMPS invocation. Clamping every run to the first of them put the end
    /// index before the start for runs belonging to a later invocation, and
    /// `lines[start..end]` panicked with
    /// `slice index starts at N but ends at M`.
    #[test]
    fn appended_restart_log_keeps_runs_of_every_invocation() {
        let text = "\
LAMMPS (30 Mar 2026)
Per MPI rank memory allocation (min/avg/max) = 1 | 1 | 1 Mbytes
   Step          Temp
         0   400
      1000   398
Loop time of 1.0 on 1 procs for 1000 steps with 10 atoms
Total wall time: 0:00:01
Per MPI rank memory allocation (min/avg/max) = 1 | 1 | 1 Mbytes
   Step          Temp
      1000   398
      2000   396
Loop time of 1.0 on 1 procs for 1000 steps with 10 atoms
Total wall time: 0:00:02
";
        let log = read_lammps_log(text.as_bytes(), "default").unwrap();
        assert_eq!(log.runs.len(), 2, "both invocations' runs survive");
        for run in &log.runs {
            let thermo = run.thermo.as_ref().expect("thermo");
            assert_eq!(thermo.n_rows(), 2);
        }
        // The second run's rows come from the second invocation, not the first.
        assert_eq!(log.runs[1].thermo.as_ref().unwrap().rows[0][0], 1000.0);
    }

    #[test]
    fn run_ranges_never_invert_when_wall_time_precedes_a_run() {
        let lines = vec![
            "Per MPI rank memory allocation (min/avg/max) = 1 | 1 | 1 Mbytes",
            "Total wall time: 0:00:01",
            "Per MPI rank memory allocation (min/avg/max) = 1 | 1 | 1 Mbytes",
        ];
        for (start, end) in find_run_ranges(&lines) {
            assert!(start <= end, "range {start}..{end} inverted");
        }
    }
}

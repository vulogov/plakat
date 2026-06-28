//! Shared system-memory gauges (RFC TUI-1 §7/§16) — a RAM + Swap bar used by the
//! Models screen and the Chat screen. Heavy swap signals an over-budget load; on
//! unified-memory Macs that's the early-warning before a Metal OOM.

use ratatui::{
    Frame,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Style},
    widgets::{Block, Borders, Gauge},
};

/// Render the RAM (left) + Swap (right) gauges into `area` (expects height ≥ 3).
pub fn render_memory_bar(f: &mut Frame, area: Rect) {
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(64), Constraint::Percentage(36)])
        .split(area);

    // RAM (unified memory on Apple Silicon).
    let total = crate::hw::total_ram_gb().max(0.1);
    let used = (total - crate::hw::available_ram_gb()).clamp(0.0, total);
    let ram = Gauge::default()
        .block(Block::default().borders(Borders::ALL).title(" RAM (unified) "))
        .gauge_style(Style::new().fg(Color::Cyan))
        .ratio((used / total).clamp(0.0, 1.0))
        .label(format!("{used:.1} / {total:.1} GB"));
    f.render_widget(ram, cols[0]);

    // Swap — heavy swap signals memory pressure / an over-budget load.
    let (swap_used, swap_total) = crate::hw::swap_gb();
    let (swap_ratio, swap_label) = if swap_total > 0.05 {
        ((swap_used / swap_total).clamp(0.0, 1.0), format!("{swap_used:.1} / {swap_total:.1} GB"))
    } else {
        (0.0, "off".to_string())
    };
    let swap_color = if swap_used > 1.0 {
        Color::Red
    } else if swap_used > 0.1 {
        Color::Yellow
    } else {
        Color::Magenta
    };
    let swap = Gauge::default()
        .block(Block::default().borders(Borders::ALL).title(" Swap "))
        .gauge_style(Style::new().fg(swap_color))
        .ratio(swap_ratio)
        .label(swap_label);
    f.render_widget(swap, cols[1]);
}

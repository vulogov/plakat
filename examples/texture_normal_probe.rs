//! G0.4 (RFC TEXTURE-1) — normal-map derivation correctness. Derives a tangent-space normal map from a
//! known height field via a Sobel gradient and checks it against the *analytic* normal, confirming the
//! encoding (unit vectors, +Z, and the OpenGL/DirectX Y convention). De-risks the B1 derivation core.
//!
//!   cargo run --release --example texture_normal_probe
//!
//! Fixture: a paraboloid h(u,v) = k(u²+v²) over u,v ∈ [-1,1], whose gradient (and thus normal) is known
//! in closed form. Pure Rust, no weights — the derivation is deterministic math.

const N: usize = 128;
const K: f32 = 0.35;

fn u(i: usize) -> f32 {
    (i as f32 / (N - 1) as f32) * 2.0 - 1.0
}

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let m = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-9);
    [v[0] / m, v[1] / m, v[2] / m]
}

fn main() {
    // Height field + its analytic per-pixel gradient (dh/di, dh/dj).
    let mut h = vec![0f32; N * N];
    let du = 2.0 / (N - 1) as f32; // pixel spacing in u-units
    for j in 0..N {
        for i in 0..N {
            h[j * N + i] = K * (u(i) * u(i) + u(j) * u(j));
        }
    }
    // analytic per-pixel gradient: dh/di = (dh/du)(du/di) = (2K·u_i)·du
    let grad_a = |i: usize, j: usize| -> (f32, f32) { (2.0 * K * u(i) * du, 2.0 * K * u(j) * du) };

    // Sobel per-pixel gradient (÷8) — what B1 will use (noise-robust central difference).
    let at = |i: i32, jj: i32| h[(jj.clamp(0, N as i32 - 1) as usize) * N + i.clamp(0, N as i32 - 1) as usize];
    let sobel = |i: usize, j: usize| -> (f32, f32) {
        let (i, j) = (i as i32, j as i32);
        let gx = (at(i + 1, j - 1) + 2.0 * at(i + 1, j) + at(i + 1, j + 1)
            - at(i - 1, j - 1) - 2.0 * at(i - 1, j) - at(i - 1, j + 1))
            / 8.0;
        let gy = (at(i - 1, j + 1) + 2.0 * at(i, j + 1) + at(i + 1, j + 1)
            - at(i - 1, j - 1) - 2.0 * at(i, j - 1) - at(i + 1, j - 1))
            / 8.0;
        (gx, gy)
    };

    // Compare normals on the interior (the paraboloid isn't periodic; B1 uses a *circular* Sobel for
    // the real, tileable case).
    let (mut sum_deg, mut max_deg, mut count) = (0f64, 0f32, 0u32);
    let (mut all_unit, mut all_pos_z) = (true, true);
    for j in 2..N - 2 {
        for i in 2..N - 2 {
            let (ax, ay) = grad_a(i, j);
            let (sx, sy) = sobel(i, j);
            // tangent-space normal: n = normalize(-dh/dx, -dh/dy, 1), OpenGL (+Y).
            let na = normalize([-ax, -ay, 1.0]);
            let ns = normalize([-sx, -sy, 1.0]);
            let dot = (na[0] * ns[0] + na[1] * ns[1] + na[2] * ns[2]).clamp(-1.0, 1.0);
            let deg = dot.acos().to_degrees();
            sum_deg += deg as f64;
            max_deg = max_deg.max(deg);
            count += 1;
            let len = (ns[0] * ns[0] + ns[1] * ns[1] + ns[2] * ns[2]).sqrt();
            if (len - 1.0).abs() > 1e-4 {
                all_unit = false;
            }
            if ns[2] <= 0.0 {
                all_pos_z = false;
            }
        }
    }
    let mean_deg = sum_deg / count as f64;

    // Encode to [0,1] RGB the way B1 will, and confirm the OpenGL/DirectX Y flip.
    let enc = |n: [f32; 3]| [(n[0] * 0.5 + 0.5), (n[1] * 0.5 + 0.5), (n[2] * 0.5 + 0.5)];
    let (sx, sy) = sobel(N - 6, N - 6); // a steep-slope point (near the paraboloid rim)
    let n_ogl = normalize([-sx, -sy, 1.0]);
    let n_dx = [n_ogl[0], -n_ogl[1], n_ogl[2]];
    let (rgb_ogl, rgb_dx) = (enc(n_ogl), enc(n_dx));

    println!("G0.4 normal-derivation correctness (paraboloid fixture, {N}×{N})\n");
    println!("  mean angular error vs analytic: {mean_deg:.3}°   (max {max_deg:.3}°)");
    println!("  unit-length normals: {all_unit}   ·   all +Z: {all_pos_z}");
    println!("  flat-point normal ≈ (0,0,1) → RGB (0.5,0.5,1.0): {:?}", enc(normalize([0.0, 0.0, 1.0])));
    println!("  OpenGL(+Y) vs DirectX(-Y) at a sloped point: G {:.3} vs {:.3}", rgb_ogl[1], rgb_dx[1]);

    // The Y-flip is correct iff the two G values are mirror images about 0.5 (opposite signs, equal
    // magnitude) at a genuinely sloped point — not an arbitrary magnitude (this bump is gentle).
    let (og, dg) = (rgb_ogl[1] - 0.5, rgb_dx[1] - 0.5);
    let flip_ok = og.signum() != dg.signum() && (og + dg).abs() < 1e-4 && og.abs() > 1e-3;
    let pass = mean_deg < 2.0 && max_deg < 6.0 && all_unit && all_pos_z && flip_ok;
    println!("\n  G0.4 exit: {}", if pass { "PASS — Sobel→tangent-space normal matches analytic; encoding + Y-convention correct" } else { "REVIEW — see numbers above" });
}

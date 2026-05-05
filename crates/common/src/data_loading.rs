// —— `.fvecs` helpers (record layout: LE `i32` dim + `dim` × LE `f32` per `benches/sift1m.rs`) —

#[inline]
fn fvecs_stride(dim: usize) -> usize {
    4 + dim * 4
}

/// Read `i32` vector dimension at `offset` (little-endian).
#[inline]
pub fn read_fvecs_dim_le(data: &[u8], offset: usize) -> Option<usize> {
    let end = offset.checked_add(4)?;
    if end > data.len() {
        return None;
    }
    Some(i32::from_le_bytes(data[offset..end].try_into().ok()?) as usize)
}

fn copy_fvec_body_into(out: &mut [f32], data: &[u8], offset: usize, dim: usize) -> bool {
    let Some(body_mul) = dim.checked_mul(4) else {
        return false;
    };
    let Some(body_off) = offset.checked_add(4) else {
        return false;
    };
    let Some(end) = body_off.checked_add(body_mul) else {
        return false;
    };
    if dim != out.len() || end > data.len() {
        return false;
    }
    let raw = &data[body_off..end];
    for (i, c) in raw.chunks_exact(4).enumerate() {
        out[i] = f32::from_le_bytes(c.try_into().unwrap());
    }
    true
}

pub fn fvecs_vector_count(data: &[u8], dim: usize) -> usize {
    let stride = fvecs_stride(dim);
    if stride == 0 {
        return 0;
    }
    data.len() / stride
}

/// Read the `index`-th vector from a `.fvecs` buffer (`index` 0 = first record).
pub fn read_fvecs_vector_at(data: &[u8], dim: usize, index: usize) -> Option<Vec<f32>> {
    let stride = fvecs_stride(dim);
    let off = index.checked_mul(stride)?;
    if off.saturating_add(stride) > data.len() {
        return None;
    }
    if read_fvecs_dim_le(data, off) != Some(dim) {
        return None;
    }
    let mut out = vec![0.0f32; dim];
    if !copy_fvec_body_into(&mut out, data, off, dim) {
        return None;
    }
    Some(out)
}

/// Import up to `limit` complete vectors (from record 0) from a `.fvecs` buffer.
/// Row indices are `0 .. n-1`; pass them to `insert` as the first argument. Stops early on malformed
/// data or when `insert` returns `false`.
pub fn import_fvecs(
    data: &[u8],
    dim: usize,
    limit: usize,
    mut insert: impl FnMut(u64, &[f32]) -> bool,
) -> usize {
    let available = fvecs_vector_count(data, dim);
    let n = limit.min(available);
    if n == 0 {
        return 0;
    }
    let stride = fvecs_stride(dim);
    let mut buf = vec![0.0f32; dim];
    for i in 0..n {
        let off = i * stride;
        if read_fvecs_dim_le(data, off) != Some(dim) {
            return i;
        }
        if !copy_fvec_body_into(&mut buf, data, off, dim) {
            return i;
        }
        if !insert(i as u64, &buf) {
            return i;
        }
    }
    n
}

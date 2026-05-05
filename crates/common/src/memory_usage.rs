/// Returns current RSS (Resident Set Size) in kilobytes.
///
/// - **Linux:** `/proc/self/status` (`VmRSS`, already kB).
/// - **macOS:** [`libc::getrusage`] (`ru_maxrss`; treated as bytes here, converted to kB).
/// - **Other:** `0`.
pub fn rss_kb() -> u64 {
    #[cfg(target_os = "linux")]
    {
        use std::fs;
        if let Ok(status) = fs::read_to_string("/proc/self/status") {
            for line in status.lines() {
                if line.starts_with("VmRSS:") {
                    return line
                        .split_whitespace()
                        .nth(1)
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(0);
                }
            }
        }
        0
    }
    #[cfg(target_os = "macos")]
    {
        use std::mem;
        unsafe {
            let mut info: libc::rusage = mem::zeroed();
            libc::getrusage(libc::RUSAGE_SELF, &mut info);
            // macOS reports in bytes, not KB
            (info.ru_maxrss / 1024) as u64
        }
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        0
    }
}

/// Returns peak RSS in kilobytes.
/// VmPeak on Linux is the high water mark — catches spikes even if brief.
pub fn peak_rss_kb() -> u64 {
    #[cfg(target_os = "linux")]
    {
        use std::fs;
        if let Ok(status) = fs::read_to_string("/proc/self/status") {
            for line in status.lines() {
                if line.starts_with("VmPeak:") {
                    return line
                        .split_whitespace()
                        .nth(1)
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(0);
                }
            }
        }
        0
    }
    #[cfg(not(target_os = "linux"))]
    {
        0
    }
}

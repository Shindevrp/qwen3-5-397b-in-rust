use std::fs;
use std::io::{BufRead, BufReader};

#[derive(Debug, Clone, Default)]
pub struct MemoryStats {
    pub vm_rss_kb: u64,
    pub vm_size_kb: u64,
    pub vm_data_kb: u64,
    pub vm_swap_kb: u64,
    pub rss_anon_kb: u64,
    pub rss_file_kb: u64,
    pub rss_shmem_kb: u64,
}

impl MemoryStats {
    pub fn snapshot() -> Self {
        let mut stats = MemoryStats::default();
        if let Ok(content) = fs::read_to_string("/proc/self/status") {
            for line in content.lines() {
                if let Some((k, v)) = line.split_once(':') {
                    let v = v.trim().split_whitespace().next().unwrap_or("0");
                    if let Ok(n) = v.parse::<u64>() {
                        match k.trim() {
                            "VmRSS" => stats.vm_rss_kb = n,
                            "VmSize" => stats.vm_size_kb = n,
                            "VmData" => stats.vm_data_kb = n,
                            "VmSwap" => stats.vm_swap_kb = n,
                            _ => {}
                        }
                    }
                }
            }
        }
        if let Ok(file) = fs::File::open("/proc/self/smaps_rollup") {
            let reader = BufReader::new(file);
            for line in reader.lines().flatten() {
                if let Some((k, v)) = line.split_once(':') {
                    let v = v.trim().split_whitespace().next().unwrap_or("0");
                    if let Ok(n) = v.parse::<u64>() {
                        match k.trim() {
                            "RssAnon" => stats.rss_anon_kb = n,
                            "RssFile" => stats.rss_file_kb = n,
                            "RssShmem" => stats.rss_shmem_kb = n,
                            _ => {}
                        }
                    }
                }
            }
        }
        stats
    }

    pub fn log(prefix: &str) {
        let s = Self::snapshot();
        eprintln!(
            "[memory {}] VmRSS={}KB VmSize={}KB VmData={}KB VmSwap={}KB RssAnon={}KB RssFile={}KB RssShmem={}KB",
            prefix,
            s.vm_rss_kb,
            s.vm_size_kb,
            s.vm_data_kb,
            s.vm_swap_kb,
            s.rss_anon_kb,
            s.rss_file_kb,
            s.rss_shmem_kb
        );
    }
}

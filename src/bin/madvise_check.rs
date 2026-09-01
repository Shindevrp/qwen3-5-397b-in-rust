use memmap2::{Mmap, UncheckedAdvice};
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;

fn rss_kb() -> u64 {
    let s = std::fs::read_to_string("/proc/self/status").unwrap();
    for line in s.lines() {
        if line.starts_with("VmRSS:") {
            if let Some(v) = line.split_whitespace().nth(1) {
                return v.parse().unwrap_or(0);
            }
        }
    }
    0
}

fn main() {
    let path = PathBuf::from("/tmp/madvise_test.bin");
    let size = 100 * 1024 * 1024; // 100 MiB
    let mut f = File::create(&path).unwrap();
    f.write_all(&vec![0u8; size]).unwrap();
    f.sync_all().unwrap();

    let file = File::open(&path).unwrap();
    let mmap = unsafe { Mmap::map(&file).unwrap() };
    println!("Mapped {} bytes", mmap.len());

    // Touch first 10 MiB
    let touch = &mmap[..10 * 1024 * 1024];
    let sum: u64 = touch.iter().map(|&b| b as u64).sum();
    println!("Touched, sum={sum}");
    println!("RSS before advise: {} KiB", rss_kb());

    // Advise dontneed first 10 MiB
    unsafe {
        mmap.unchecked_advise_range(UncheckedAdvice::DontNeed, 0, 10 * 1024 * 1024)
            .unwrap();
    }
    println!("Advise done");
    println!("RSS after advise: {} KiB", rss_kb());

    println!("madvise check completed");
}

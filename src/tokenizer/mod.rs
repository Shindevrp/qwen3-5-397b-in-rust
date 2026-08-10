use anyhow::Result;
use tokenizers::Tokenizer;

pub struct QwenTokenizer {
    tokenizer: Tokenizer,
}

impl QwenTokenizer {
    pub fn from_file<P: AsRef<std::path::Path>>(path: P) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(path)
            .map_err(|e| anyhow::anyhow!("failed to load tokenizer: {}", e))?;
        Ok(Self { tokenizer })
    }

    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        let encoding = self
            .tokenizer
            .encode(text, add_special_tokens)
            .map_err(|e| anyhow::anyhow!("encoding failed: {}", e))?;
        Ok(encoding.get_ids().to_vec())
    }

    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        self.tokenizer
            .decode(ids, skip_special_tokens)
            .map_err(|e| anyhow::anyhow!("decoding failed: {}", e))
    }

    pub fn vocab_size(&self) -> usize {
        self.tokenizer.get_vocab_size(true)
    }

    #[allow(dead_code)]
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.tokenizer.token_to_id(token)
    }

    #[allow(dead_code)]
    pub fn id_to_token(&self, id: u32) -> Option<String> {
        self.tokenizer.id_to_token(id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::NamedTempFile;

    fn write_temp_tokenizer(json: &str) -> NamedTempFile {
        let f = NamedTempFile::new().unwrap();
        fs::write(f.path(), json).unwrap();
        f
    }

    #[test]
    fn tokenizer_basic_encode_decode() {
        let tokenizer_json = r#"{
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [
                {"id": 0, "content": "<|endoftext|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
            ],
            "normalizer": {
                "type": "Sequence",
                "normalizers": [
                    {"type": "NFD"},
                    {"type": "Lowercase"},
                    {"type": "StripAccents"}
                ]
            },
            "pre_tokenizer": {
                "type": "ByteLevel",
                "add_prefix_space": false,
                "trim_offsets": true,
                "use_regex": true
            },
            "model": {
                "type": "BPE",
                "dropout": null,
                "unk_token": "<|endoftext|>",
                "continuing_subword_prefix": "",
                "end_of_word_suffix": "",
                "fuse_unk": true,
                "vocab": {
                    "a": 1,
                    "b": 2,
                    "c": 3,
                    "ab": 4,
                    "abc": 5,
                    "Ġhello": 6,
                    "Ġworld": 7,
                    "Ġ": 8
                },
                "merges": [
                    "a b",
                    "ab c"
                ]
            },
            "post_processor": null,
            "decoder": {
                "type": "ByteLevel",
                "add_prefix_space": false,
                "trim_offsets": true,
                "use_regex": true
            }
        }"#;

        let f = write_temp_tokenizer(tokenizer_json);
        let tok = QwenTokenizer::from_file(f.path()).unwrap();

        assert_eq!(tok.vocab_size(), 9);

        let ids = tok.encode("abc", false).unwrap();
        assert_eq!(ids, vec![5]);

        let ids = tok.encode("a b c", false).unwrap();
        assert_eq!(ids, vec![1, 8, 2, 8, 3]);

        let decoded = tok.decode(&[6, 7], true).unwrap();
        // ByteLevel decoder adds leading space
        assert_eq!(decoded, " hello world");
    }

    #[test]
    fn tokenizer_special_tokens() {
        let tokenizer_json = r#"{
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [
                {"id": 0, "content": "<|endoftext|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
                {"id": 1, "content": "图", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
                {"id": 2, "content": "文", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
            ],
            "normalizer": {
                "type": "Sequence",
                "normalizers": [
                    {"type": "NFD"},
                    {"type": "Lowercase"},
                    {"type": "StripAccents"}
                ]
            },
            "pre_tokenizer": {
                "type": "ByteLevel",
                "add_prefix_space": false,
                "trim_offsets": true,
                "use_regex": true
            },
            "model": {
                "type": "BPE",
                "dropout": null,
                "unk_token": "<|endoftext|>",
                "continuing_subword_prefix": "",
                "end_of_word_suffix": "",
                "fuse_unk": true,
                "vocab": {
                    "a": 3,
                    "b": 4,
                    "图": 1,
                    "文": 2
                },
                "merges": []
            },
            "post_processor": null,
            "decoder": {
                "type": "ByteLevel",
                "add_prefix_space": false,
                "trim_offsets": true,
                "use_regex": true
            }
        }"#;

        let f = write_temp_tokenizer(tokenizer_json);
        let tok = QwenTokenizer::from_file(f.path()).unwrap();

        let eos_id = tok.token_to_id("<|endoftext|>").unwrap();
        let img_id = tok.token_to_id("图").unwrap();
        let vid_id = tok.token_to_id("文").unwrap();
        assert_ne!(eos_id, img_id);
        assert_ne!(eos_id, vid_id);
        assert_ne!(img_id, vid_id);

        let ids = tok.encode("图文", false).unwrap();
        assert_eq!(ids, vec![img_id, vid_id]);
    }
}

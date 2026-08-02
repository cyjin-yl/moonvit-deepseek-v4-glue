# Vendor directory

Third-party model code vendored for self-contained operation, so the vision
tower can be built and loaded without downloading a full multi-terabyte
model repository or relying on `trust_remote_code` fetches at runtime.

- `kimi_k3/` — MoonViT-V2 vision tower extracted from
  [moonshotai/Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3)
  (Apache-2.0-derived llava parts + Kimi K3 License; see
  `kimi_k3/LICENSE`). Only vision-related modules are kept; the text-model
  dependency (`modeling_kimi_linear`) and the conditional-generation class
  were removed.

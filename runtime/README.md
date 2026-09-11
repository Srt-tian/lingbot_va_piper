# Piper inference runtime snapshot

These 29 files are copied without modification from the existing Piper inference working tree, based on commit `0cd690a4f02bba4f611e7e1aedc3f7eeee8ed315` plus local client changes. Each file is pinned by SHA256 in `../integration/reference_manifest.json`. The wrapper validates them before installing its adapter.

This directory includes the project's hardware/runtime code and its OpenPI client integration. Preserve existing file copyright notices. The top-level LingBot license documents the upstream model code; this source snapshot is not a claim that all independently developed runtime components originated in Robbyant/LingBot-VA. Piper SDK and RealSense packages are external dependencies and are not vendored here.

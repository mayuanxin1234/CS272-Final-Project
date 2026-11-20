# Lab 1/2 Colab bundle

Minimum shared files for both required Lecture 1 Colab notebooks: the `lab1`
package, CPU-fast and strong configurations, 5,000- and 20,000-transition
datasets, quick and strong world-model checkpoints, and training histories.
Lab 1 demonstrates five-epoch learning; Lab 2 uses the separately supplied
20-epoch model by default.

Both checkpoints are inference copies: optimizer state was removed and all
model tensors are identical to the originals (see `MANIFEST.json`). The
datasets were recorded in HighwayEnv (Farama Foundation, MIT license). See
`LICENSE` and `THIRD_PARTY_NOTICES.md`.

Neither notebook collects data.

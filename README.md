# Koinon

<p align="center" width="100%">
    <img src="./psyche-book/src/psyche.jpg">
</p>

Koinon turns distributed, heterogeneous compute into resources that can be
discovered, invoked, measured, and eventually traded without taking control
away from the people operating the hardware.

The system is organized as one stack with distinct responsibilities:

- **Koinon** is the network and execution core: identity, discovery, routing,
  inference protocols, and node resource control.
- **TTinker Grid** is the economic coordination layer: supply, demand,
  measurement, reputation, pricing, and settlement.
- **Psyche** supplies the existing distributed training systems and networking
  foundations. Existing `psyche-*` package names remain unchanged for
  compatibility.

The current product milestone is a consent-first inference path from an
OpenAI-compatible gateway, over outbound P2P, to a user-operated local model.
Installation never implies permission to use a GPU.

## Repository map

- [`architectures/inference-only`](architectures/inference-only) contains the
  current inference node and gateway.
- [`shared/inference`](shared/inference) contains inference protocol and
  execution building blocks.
- [`services/marketplace`](services/marketplace) contains the TTinker Grid
  marketplace prototype and compatibility control plane.
- [`psyche-book`](psyche-book) and the existing training crates contain the
  Psyche documentation and distributed training implementation.

## Project direction

- [Vision](docs/vision.md)
- [Current roadmap](docs/roadmap.md)
- [Approved Koinon Grid architecture design](docs/superpowers/specs/2026-07-30-koinon-grid-vision-design.md)

Current capabilities and future claims are kept separate in the roadmap. In
particular, the marketplace currently reports inference as unavailable and all
displayed quotes remain simulations.

For detailed documentation on the existing Psyche training project, visit the
[Psyche docs](https://docs.psyche.network).

<p align="center" width="100%">
    <a href="https://www.youtube.com/watch?v=XMWI3nDk48c">
        <img src="./psyche-book/src/psyche_youtube.png">
    </a>
</p>

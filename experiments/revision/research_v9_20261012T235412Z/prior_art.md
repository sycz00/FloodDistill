# Claim boundary and checked primary references

[Prithvi-EO-2.0](https://arxiv.org/abs/2412.02732) is the pretrained backbone reference. Its current paper version is v3 (6 March2026); the local experiment instead pins the actual task-adapted checkpoint bytes. Token patch size is distinct from the inspected downstream decoder output stride.

[PANGAEA](https://arxiv.org/abs/2412.04204) compares geospatial foundation models with supervised segmentation baselines and finds that foundation models do not consistently outperform them. The present experiment tests one fixed task-adapted teacher and does not establish general foundation-model necessity.

[SegFix](https://arxiv.org/abs/2007.04269) already studies model-agnostic boundary refinement. Its learned boundary/interior correspondence differs from this fixed MNDWI re-decision, but boundary refinement itself is established. Pseudo-label refinement, guided image filtering and spectral water indices are established ingredients; this follow-up makes no first-in-literature claim.

[WorldFloods-v2 paper](https://www.nature.com/articles/s41598-023-47595-7) describes a cloud-aware optical flood-mapping framework and dataset. Local native-window/cloud/permanent-water handling is frozen separately and verified against retained evaluator code. Mixed reference satellite/date-lag groups and conservative overlap exclusions limit the interpretation of a teacher-edge hypothesis.

Primary pages checked13October2026. These anchors are context, not exhaustive novelty clearance. Any contribution claim depends on the complete paired controls and declared sealed-test gates in frozen_protocol.md.

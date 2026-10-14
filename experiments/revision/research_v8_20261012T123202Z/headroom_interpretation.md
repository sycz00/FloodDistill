# Teacher headroom

Observed pooled correction and harm rates below. They do not select any deployable threshold.

- s11, seed 42: G=0.007577, H=0.008356; oracle water IoU=0.875581720726167; remaining both-wrong pixels=332583.

- s11, seed 64: G=0.005736, H=0.006994; oracle water IoU=0.8659556938305681; remaining both-wrong pixels=360532.

- source_validation, seed 42: G=0.009563, H=0.006485; oracle water IoU=0.8658035853843287; remaining both-wrong pixels=312515.

- source_validation, seed 64: G=0.010714, H=0.006099; oracle water IoU=0.8633261455939613; remaining both-wrong pixels=320337.

- sturm, seed 42: G=0.013689, H=0.016780; oracle water IoU=0.6863924099235538; remaining both-wrong pixels=64096019.

- sturm, seed 64: G=0.010761, H=0.014719; oracle water IoU=0.6823037515500593; remaining both-wrong pixels=65486735.

- wf, seed 42: G=0.006776, H=0.009533; oracle water IoU=0.6812113885406285; remaining both-wrong pixels=10666651.

- wf, seed 64: G=0.005812, H=0.010101; oracle water IoU=0.6836498833165533; remaining both-wrong pixels=10585812.

Inspect every event and class in event_headroom.csv. A pooled oracle improvement measures complementary information, not whether a frozen linear head or deployable router can access it.

Reliability-proxy errors on actual support reference labels are offline diagnostics in support_cache/manifest.json; they never fit a reliability model.

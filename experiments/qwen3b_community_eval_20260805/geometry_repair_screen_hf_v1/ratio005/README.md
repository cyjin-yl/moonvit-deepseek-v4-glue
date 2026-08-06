# Package 15P ratio005 high-frequency trajectory

`ratio005` is the pre-registered 5% auxiliary-gradient arm
(`lambda=0.01018730507868909`). It uses the exact same model, cached feature
order, health probe, generation-independent teacher-forced probe, optimizer
budget and stop thresholds as `control`.

The arm reached optimizer step 2 and stopped for the same two adverse-trend
guards. Projector and receiver geometry therefore collapse in the same
step-1-to-2 interval; the small geometry term changes total loss at step 2 to
2.45268 but does not preserve the representation. This is a failed geometry
screen, with no visual-capability claim and no checkpoint promotion.

The complete raw checkpoint/optimizer copy is bound in
`RAW_ARTIFACT_POINTER.json`; the large files remain outside Git.

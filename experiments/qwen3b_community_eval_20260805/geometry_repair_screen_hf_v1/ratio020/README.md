# Package 15P ratio020 high-frequency trajectory

`ratio020` is the pre-registered 20% auxiliary-gradient arm
(`lambda=0.04074922031475636`). It follows the same frozen data order,
projector initialization, 50-image probe and auto-stop contract as control and
`ratio005`.

The arm stopped at optimizer step 2 with the same collapse interval `[1,2]`.
The geometry term raised total loss to 2.49668, yet receiver rank/spread
ratios still fell to 0.3623/0.2255. There is no capability claim or checkpoint
promotion. Full raw checkpoints and optimizer state are bound by the pointer.

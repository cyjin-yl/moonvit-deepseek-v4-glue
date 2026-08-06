# Package 15P ratio080 high-frequency trajectory

`ratio080` is the strongest pre-registered auxiliary-gradient arm
(`lambda=0.16299688125902545`). It uses the same frozen initialization, record
order, probe set and auto-stop thresholds as the other three arms.

It stops at optimizer step 2 with onset `[1,2]`. Total loss is 2.67265, while
receiver spread/rank ratios fall to 0.2258/0.3628. The full lambda screen
therefore has no passing arm; the 500-step expansion is cancelled and the next
experiment must change projector structure or update scale.

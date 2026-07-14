## ZSC SP/XP Performance

| method | coord_ring SP | coord_ring XP | counter_circuit SP | counter_circuit XP | cramped_room5x5 SP | cramped_room5x5 XP | asymm_advantages SP | asymm_advantages XP | forced_coord SP | forced_coord XP | Average SP | Average XP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lmpred | $297 \pm 3$ | $283 \pm 3$ | $237 \pm 15$ | $191 \pm 11$ | $236 \pm 2$ | $223 \pm 4$ |  |  |  |  | $257 \pm 5$ | $233 \pm 4$ |
| lmpred_ablate | $295 \pm 7$ | $201 \pm 20$ | $250 \pm 10$ | $241 \pm 4$ | $226 \pm 10$ | $198 \pm 10$ |  |  |  |  | $257 \pm 5$ | $213 \pm 8$ |
| lmpred_gamma0 | $288 \pm 11$ | $257 \pm 5$ | $187 \pm 41$ | $167 \pm 13$ | $235 \pm 4$ | $217 \pm 4$ | $454 \pm 9$ | $414 \pm 16$ | $300 \pm 10$ | $115 \pm 27$ | $293 \pm 9$ | $234 \pm 7$ |
| lmpred_gamma0_ablate | $298 \pm 11$ | $263 \pm 4$ | $205 \pm 19$ | $142 \pm 11$ | $224 \pm 11$ | $199 \pm 7$ | $467 \pm 7$ | $457 \pm 6$ | $298 \pm 6$ | $153 \pm 24$ | $298 \pm 5$ | $243 \pm 6$ |
| lmpred_gamma09 | $282 \pm 11$ | $275 \pm 4$ | $192 \pm 22$ | $158 \pm 14$ | $230 \pm 6$ | $175 \pm 14$ | $468 \pm 10$ | $425 \pm 19$ | $277 \pm 13$ | $158 \pm 22$ | $290 \pm 6$ | $238 \pm 7$ |
| lmpred_gamma09_ablate | $282 \pm 22$ | $232 \pm 17$ | $240 \pm 13$ | $215 \pm 6$ | $228 \pm 5$ | $191 \pm 8$ | $451 \pm 9$ | $448 \pm 5$ | $314 \pm 3$ | $140 \pm 30$ | $303 \pm 6$ | $245 \pm 7$ |
| lmpred_no_self_pred | $261 \pm 7$ | $254 \pm 7$ | $198 \pm 19$ | $154 \pm 8$ | $227 \pm 4$ | $220 \pm 3$ | $468 \pm 6$ | $445 \pm 7$ | $295 \pm 10$ | $140 \pm 22$ | $290 \pm 5$ | $243 \pm 5$ |
| lmpred_ablate_no_self_pred | $284 \pm 23$ | $204 \pm 19$ | $207 \pm 20$ | $144 \pm 13$ | $232 \pm 3$ | $223 \pm 2$ | $445 \pm 16$ | $376 \pm 23$ | $295 \pm 6$ | $216 \pm 17$ | $293 \pm 7$ | $232 \pm 7$ |

## ZSC Alignment MSE

| method | coord_ring | counter_circuit | cramped_room5x5 | asymm_advantages | forced_coord | Average |
| --- | --- | --- | --- | --- | --- | --- |
| lmpred | $0.1384 \pm 0.0080$ | $0.1865 \pm 0.0086$ | $0.0813 \pm 0.0103$ |  |  | $0.1354 \pm 0.0052$ |
| lmpred_ablate | $0.1387 \pm 0.0104$ | $0.1662 \pm 0.0079$ | $0.1034 \pm 0.0109$ |  |  | $0.1361 \pm 0.0057$ |
| lmpred_gamma0 | $0.1939 \pm 0.0273$ | $0.2146 \pm 0.0098$ | $0.0988 \pm 0.0131$ | $0.0874 \pm 0.0096$ | $0.0509 \pm 0.0057$ | $0.1291 \pm 0.0068$ |
| lmpred_gamma0_ablate | $0.1842 \pm 0.0172$ | $0.2322 \pm 0.0074$ | $0.1068 \pm 0.0131$ | $0.1037 \pm 0.0211$ | $0.0588 \pm 0.0062$ | $0.1371 \pm 0.0064$ |
| lmpred_gamma09 | $0.1584 \pm 0.0098$ | $0.2419 \pm 0.0088$ | $0.0885 \pm 0.0081$ | $0.0772 \pm 0.0070$ | $0.0578 \pm 0.0035$ | $0.1248 \pm 0.0035$ |
| lmpred_gamma09_ablate | $0.1675 \pm 0.0186$ | $0.2220 \pm 0.0130$ | $0.1067 \pm 0.0151$ | $0.0843 \pm 0.0053$ | $0.0577 \pm 0.0041$ | $0.1276 \pm 0.0056$ |
| lmpred_no_self_pred | $0.1543 \pm 0.0100$ | $0.2030 \pm 0.0110$ | $0.0972 \pm 0.0084$ | $0.1057 \pm 0.0103$ | $0.0855 \pm 0.0107$ | $0.1292 \pm 0.0045$ |
| lmpred_ablate_no_self_pred | $0.1698 \pm 0.0196$ | $0.2044 \pm 0.0094$ | $0.1051 \pm 0.0124$ | $0.0793 \pm 0.0090$ | $0.0725 \pm 0.0057$ | $0.1262 \pm 0.0054$ |

## Ad-Hoc Teamplay Performance

| method | coord_ring | counter_circuit | cramped_room5x5 | asymm_advantages | forced_coord | Average |
| --- | --- | --- | --- | --- | --- | --- |
| lmpred | $257 \pm 6$ | $150 \pm 6$ | $211 \pm 3$ |  |  | $206 \pm 3$ |
| lmpred_ablate | $226 \pm 7$ | $172 \pm 6$ | $193 \pm 5$ |  |  | $197 \pm 3$ |
| lmpred_gamma0 | $236 \pm 6$ | $142 \pm 6$ | $193 \pm 5$ |  |  | $190 \pm 3$ |
| lmpred_gamma0_ablate | $245 \pm 6$ | $141 \pm 6$ | $185 \pm 5$ |  |  | $191 \pm 3$ |
| lmpred_gamma09 | $255 \pm 6$ | $146 \pm 6$ | $178 \pm 6$ |  |  | $193 \pm 3$ |
| lmpred_gamma09_ablate | $223 \pm 7$ | $162 \pm 5$ | $191 \pm 5$ |  |  | $192 \pm 3$ |
| lmpred_no_self_pred | $235 \pm 6$ | $132 \pm 5$ | $192 \pm 5$ |  |  | $186 \pm 3$ |
| lmpred_ablate_no_self_pred | $232 \pm 7$ | $140 \pm 6$ | $192 \pm 5$ |  |  | $188 \pm 4$ |

## Ad-Hoc Teamplay Alignment MSE

| method | coord_ring | counter_circuit | cramped_room5x5 | asymm_advantages | forced_coord | Average |
| --- | --- | --- | --- | --- | --- | --- |
| lmpred | $0.1416 \pm 0.0061$ | $0.2020 \pm 0.0062$ | $0.0943 \pm 0.0071$ |  |  | $0.1459 \pm 0.0037$ |
| lmpred_ablate | $0.1510 \pm 0.0065$ | $0.1869 \pm 0.0063$ | $0.0908 \pm 0.0063$ |  |  | $0.1429 \pm 0.0037$ |
| lmpred_gamma0 | $0.1485 \pm 0.0084$ | $0.2140 \pm 0.0064$ | $0.1089 \pm 0.0092$ |  |  | $0.1571 \pm 0.0047$ |
| lmpred_gamma0_ablate | $0.1511 \pm 0.0103$ | $0.2211 \pm 0.0073$ | $0.0908 \pm 0.0077$ |  |  | $0.1544 \pm 0.0049$ |
| lmpred_gamma09 | $0.1558 \pm 0.0064$ | $0.2026 \pm 0.0056$ | $0.1026 \pm 0.0107$ |  |  | $0.1536 \pm 0.0045$ |
| lmpred_gamma09_ablate | $0.1513 \pm 0.0072$ | $0.2098 \pm 0.0061$ | $0.0936 \pm 0.0055$ |  |  | $0.1516 \pm 0.0036$ |
| lmpred_no_self_pred | $0.1392 \pm 0.0058$ | $0.2016 \pm 0.0074$ | $0.1010 \pm 0.0074$ |  |  | $0.1473 \pm 0.0040$ |
| lmpred_ablate_no_self_pred | $0.1616 \pm 0.0092$ | $0.2109 \pm 0.0069$ | $0.1014 \pm 0.0102$ |  |  | $0.1580 \pm 0.0051$ |

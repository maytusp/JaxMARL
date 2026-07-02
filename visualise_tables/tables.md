## ZSC SP/XP Performance

| method | coord_ring SP | coord_ring XP | counter_circuit SP | counter_circuit XP | cramped_room5x5 SP | cramped_room5x5 XP |
| --- | --- | --- | --- | --- | --- | --- |
| ph2v5 | $289 \pm 8$ | $285 \pm 2$ | $238 \pm 13$ | $186 \pm 8$ | $235 \pm 2$ | $215 \pm 4$ |
| ph2v5_ablate | $277 \pm 18$ | $235 \pm 7$ | $243 \pm 6$ | $207 \pm 8$ | $237 \pm 1$ | $227 \pm 2$ |
| ph2v4 | $276 \pm 10$ | $237 \pm 10$ | $228 \pm 29$ | $203 \pm 7$ | $232 \pm 5$ | $207 \pm 11$ |
| ph2v4_ablate | $279 \pm 15$ | $220 \pm 8$ | $194 \pm 23$ | $157 \pm 8$ | $212 \pm 17$ | $206 \pm 9$ |
| e3t | $304 \pm 5$ | $269 \pm 11$ | $223 \pm 5$ | $178 \pm 8$ | $195 \pm 11$ | $151 \pm 8$ |
| sp | $300 \pm 4$ | $280 \pm 5$ | $155 \pm 14$ | $153 \pm 5$ | $217 \pm 9$ | $134 \pm 21$ |
| lmpred_ema | $301 \pm 7$ | $246 \pm 12$ | $226 \pm 12$ | $179 \pm 8$ | $230 \pm 8$ | $217 \pm 5$ |
| lmpred_ema_ablate | $292 \pm 14$ | $208 \pm 19$ | $250 \pm 11$ | $215 \pm 9$ | $232 \pm 3$ | $222 \pm 3$ |
| lmpred_ema_gamma0 | $291 \pm 10$ | $246 \pm 10$ | $233 \pm 16$ | $214 \pm 5$ | $227 \pm 5$ | $209 \pm 3$ |
| lmpred_ema_gamma09 | $219 \pm 54$ | $230 \pm 10$ | $243 \pm 12$ | $229 \pm 5$ | $185 \pm 35$ | $191 \pm 6$ |
| lmpred_ema_no_self_pred | $294 \pm 6$ | $174 \pm 21$ | $230 \pm 18$ | $195 \pm 8$ | $224 \pm 5$ | $183 \pm 8$ |

## Ad-Hoc Teamplay Performance

| method | coord_ring | counter_circuit | cramped_room5x5 |
| --- | --- | --- | --- |
| ph2v5 | $261 \pm 6$ | $159 \pm 5$ | $210 \pm 3$ |
| ph2v5_ablate | $233 \pm 6$ | $166 \pm 5$ | $209 \pm 4$ |
| ph2v4 | $240 \pm 7$ | $162 \pm 5$ | $205 \pm 5$ |
| ph2v4_ablate | $234 \pm 6$ | $141 \pm 5$ | $201 \pm 4$ |
| e3t | $187 \pm 6$ | $124 \pm 6$ | $192 \pm 3$ |
| sp | $257 \pm 6$ | $137 \pm 4$ | $187 \pm 6$ |
| fcp | $228 \pm 6$ | $91 \pm 4$ | $180 \pm 5$ |
| mep_br | $227 \pm 6$ | $126 \pm 4$ | $207 \pm 3$ |
| pbt | $253 \pm 6$ | $146 \pm 4$ | $164 \pm 7$ |
| lmpred_ema | $238 \pm 6$ | $153 \pm 5$ | $195 \pm 5$ |
| lmpred_ema_ablate | $223 \pm 8$ | $165 \pm 5$ | $180 \pm 7$ |
| lmpred_ema_gamma0 | $236 \pm 7$ | $165 \pm 5$ | $196 \pm 4$ |
| lmpred_ema_gamma09 | $237 \pm 6$ | $170 \pm 6$ | $188 \pm 6$ |
| lmpred_ema_no_self_pred | $214 \pm 8$ | $165 \pm 5$ | $165 \pm 7$ |

## Ad-Hoc Teamplay Alignment MSE

| method | coord_ring | counter_circuit | cramped_room5x5 |
| --- | --- | --- | --- |
| ph2v5 | $0.1434 \pm 0.0076$ | $0.1914 \pm 0.0067$ | $0.0909 \pm 0.0066$ |
| ph2v5_ablate | $0.1457 \pm 0.0069$ | $0.1855 \pm 0.0061$ | $0.1027 \pm 0.0081$ |
| ph2v4 | $0.1458 \pm 0.0055$ | $0.1856 \pm 0.0052$ | $0.0930 \pm 0.0077$ |
| ph2v4_ablate | $0.1619 \pm 0.0082$ | $0.2012 \pm 0.0054$ | $0.0854 \pm 0.0058$ |
| e3t | $0.1814 \pm 0.0083$ | $0.2296 \pm 0.0068$ | $0.1505 \pm 0.0051$ |
| sp | $0.2017 \pm 0.0088$ | $0.2601 \pm 0.0067$ | $0.1130 \pm 0.0061$ |
| fcp | $0.1965 \pm 0.0072$ | $0.2738 \pm 0.0075$ | $0.1517 \pm 0.0068$ |
| mep_br | $0.1757 \pm 0.0063$ | $0.2400 \pm 0.0075$ | $0.1153 \pm 0.0055$ |
| pbt | $0.2108 \pm 0.0071$ | $0.2446 \pm 0.0049$ | $0.2399 \pm 0.0963$ |
| lmpred_ema | $0.1547 \pm 0.0091$ | $0.1900 \pm 0.0062$ | $0.1367 \pm 0.0279$ |
| lmpred_ema_ablate | $0.1571 \pm 0.0074$ | $0.1931 \pm 0.0057$ | $0.1208 \pm 0.0152$ |
| lmpred_ema_gamma0 | $0.1390 \pm 0.0094$ | $0.1890 \pm 0.0063$ | $0.0907 \pm 0.0061$ |
| lmpred_ema_gamma09 | $0.1412 \pm 0.0096$ | $0.1981 \pm 0.0082$ | $0.1130 \pm 0.0126$ |
| lmpred_ema_no_self_pred | $0.1507 \pm 0.0083$ | $0.1940 \pm 0.0060$ | $0.1299 \pm 0.0167$ |

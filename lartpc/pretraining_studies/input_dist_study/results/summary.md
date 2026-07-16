# Input-distribution study (P05F) summary

files: 415680 (0 bad), true points: 29,517,498,592

## Class counts

- electron: 586,236,504
- muon: 24,991,088,614
- pion: 151,506,832
- proton: 335,047,245
- gamma: 397,590,393
- michel: 159,422,618
- delta: 2,721,223,148
- led: 175,383,238

## Clip fractions above 1000 ADC (y-plane)

- electron: 0.0003
- muon: 0.0014
- pion: 0.0006
- proton: 0.0016
- gamma: 0.0019
- michel: 0.0001
- delta: 0.0014
- led: 0.0012

## Best transform per pair (y-plane, by aug-noise-aware d')

- muon vs pion: AUC=0.594 (transform-invariant); current log d'=0.131; best=quantile d'=0.336
- muon vs proton: AUC=0.540 (transform-invariant); current log d'=0.058; best=quantile d'=0.135
- pion vs proton: AUC=0.626 (transform-invariant); current log d'=0.188; best=quantile d'=0.467
- electron vs gamma: AUC=0.512 (transform-invariant); current log d'=0.008; best=linear1000 d'=0.047

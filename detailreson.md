# Detailed Reasons From EDA

This file explains the results in [final_metrics.xlsx](/f:/files/work/btp/simpletm/simpleTMG/final_metrics.xlsx) using the evidence from [eda-analysis.ipynb](/f:/files/work/btp/simpletm/simpleTMG/eda-analysis.ipynb). The runs were produced by [simpletmg-alldataset.ipynb](/f:/files/work/btp/simpletm/simpleTMG/simpletmg-alldataset.ipynb).

## Scope Of The Explanation

The result workbook contains these eight datasets:

- `ETTh1`
- `ETTm1`
- `PEMS03`
- `Solar`
- `electricity`
- `smartbuilding`
- `traffic`
- `weather`

The notebook compares four variants on each dataset:

- `SWT_original`
- `SWT_dual`
- `FFT_original`
- `FFT_dual`

So every explanation below answers two questions:

1. Why did `SWT` beat or lose to `FFT`?
2. Why did `dual` attention beat or lose to `original` attention?

## Experimental Context From The Run Notebook

From [simpletmg-alldataset.ipynb](/f:/files/work/btp/simpletm/simpleTMG/simpletmg-alldataset.ipynb):

- the same four variants were run on every enabled dataset
- `seq_len = 96`
- `pred_len = 96`
- `e_layers = 1`
- seed fixed to `2025`
- each dataset kept its own `enc_in`, `d_model`, `d_ff`, `wv`, `alpha`, `batch_size`, and `freq`
- the comparison is therefore a controlled ablation inside the same SimpleTMG backbone

This matters because the ranking should be interpreted as:

- "which tokenization and attention fusion works better under the same backbone?"

not as:

- "which dataset is globally easier?"

## How The EDA Is Used

The EDA notebook gives these signals:

- `MeanAbsCorr`: if high, the model should benefit from multivariate/channel interaction.
- `AutocorrHalfLife`: if high, future values depend on a longer temporal context.
- `TemporalProfileStrength`: if high, repeated daily structure exists and temporal attention can help.
- `SpectralEntropy`: if low, the signal is concentrated in a few frequencies and FFT should have a better chance.
- `WaveletDetailPct`: if high, the signal has stronger local multi-scale detail and SWT should have a better chance.
- `LocalVarianceRatio`: if high, the data change sharply at short range, again helping SWT more than a purely global spectral view.

For ranking the variants, `MSE` is used as the primary metric because it is the most standard overall selection criterion in the workbook. `MAE` and `R2` are used as supporting evidence.

## Overall Pattern

The workbook shows:

- `SWT_dual` wins on `ETTh1`, `PEMS03`, `electricity`, `smartbuilding`, and `traffic`
- `SWT_original` wins on `ETTm1` and `Solar`
- `FFT_dual` wins on `weather`
- `FFT_original` does not win on any of the eight reported datasets

This is already consistent with the EDA at a high level:

- datasets with stronger local detail and local variability tend to favor `SWT`
- datasets with smooth long-range periodic structure favor `FFT`
- the dual branch helps when there is enough usable temporal structure beyond what the original channel/scale path already captures

## Summary Table

| Dataset | Best Variant | Second Best | Best MSE | Gap To Second | Main EDA Reason |
| --- | --- | --- | ---: | ---: | --- |
| `ETTh1` | `SWT_dual` | `FFT_dual` | `0.4465` | `0.53%` | repeated daily structure plus non-trivial local variation |
| `ETTm1` | `SWT_original` | `FFT_dual` | `0.3497` | `2.05%` | smoother, persistent signal where extra fusion is not needed |
| `PEMS03` | `SWT_dual` | `SWT_original` | `1151.84` | `0.18%` | very strong multivariate coupling, dual helps slightly, FFT loses badly |
| `Solar` | `SWT_original` | `SWT_dual` | `0.1710` | `3.62%` | strong periodicity but sharp day-night transitions and many zeros |
| `electricity` | `SWT_dual` | `FFT_dual` | `0.1935` | `0.12%` | diffuse spectrum, moderate local detail, small gain from temporal branch |
| `smartbuilding` | `SWT_dual` | `FFT_dual` | `0.5622` | `0.18%` | strongest mix of daily pattern plus local on/off transitions |
| `traffic` | `SWT_dual` | `FFT_dual` | `0.5334` | `4.96%` | highest local variance and wavelet detail, so SWT clearly fits |
| `weather` | `FFT_dual` | `SWT_dual` | `0.1916` | `0.42%` | smoothest and most spectrally concentrated dataset in the table |

## Dataset-Wise Detailed Interpretation

### `ETTh1`

Ranking by `MSE`:

1. `SWT_dual` = `0.4465`
2. `FFT_dual` = `0.4488`
3. `SWT_original` = `0.4513`
4. `FFT_original` = `0.4571`

Relevant EDA signals:

- `MeanAbsCorr = 0.222`, so channel dependence exists but is not extreme
- `AutocorrHalfLife = 6`, so persistence is short
- `TemporalProfileStrength = 0.661`, which is strong
- `SpectralEntropy = 0.584`, so the spectrum is not especially clean for FFT
- `WaveletDetailPct = 8.08`
- `LocalVarianceRatio = 0.368`

Why `SWT` beats `FFT`:

`ETTh1` is not a purely smooth periodic signal. It has visible daily repetition, but it also has enough short-range change and local detail that preserving locality matters. Since `SpectralEntropy` is only medium and not particularly low, FFT does not get a strong global periodicity advantage. SWT keeps local structure, so it handles those short-term changes a little better.

Why `dual` beats `original`:

The daily profile strength is high even though the half-life is short. That means there is useful temporal structure, but it is more recurrent and local than extremely long-range. The additional temporal branch helps the model exploit this repeated pattern. The gain is small, which is exactly what the EDA suggests: temporal information is useful, but not dominant enough to create a large jump.

Why the final ranking makes sense:

This is a balanced dataset where neither FFT nor the temporal branch should dominate completely. That is why the margins are small. The winner is the model that captures both repeated daily behavior and local detail at the same time: `SWT_dual`.

### `ETTm1`

Ranking by `MSE`:

1. `SWT_original` = `0.3497`
2. `FFT_dual` = `0.3570`
3. `SWT_dual` = `0.3583`
4. `FFT_original` = `0.3630`

Relevant EDA signals:

- `MeanAbsCorr = 0.224`
- `AutocorrHalfLife = 29`
- `TemporalProfileStrength = 0.658`
- `SpectralEntropy = 0.464`
- `WaveletDetailPct = 1.20`
- `LocalVarianceRatio = 0.090`

Why `SWT` still wins over `FFT`:

`ETTm1` is smoother than `ETTh1`, and its spectrum is somewhat more concentrated, so FFT becomes more competitive here. That is why the top three variants are closer together. However, the signal still does not have an overwhelming FFT advantage, and SWT remains slightly stronger under this fixed backbone.

Why `original` beats `dual`:

This is the most important point for `ETTm1`. The EDA says the series is persistent and repetitive, but it is also smooth and low-detail. In that situation, the base model already captures much of the useful structure. The extra temporal branch does not add enough new information, so the added fusion complexity slightly hurts rather than helps.

Why the final ranking makes sense:

`ETTm1` looks like a dataset where the backbone is already close to sufficient. That is why the simplest strong option, `SWT_original`, wins. The result is consistent with the EDA because the signal is stable, smooth, and not dominated by sharp local events.

### `PEMS03`

Ranking by `MSE`:

1. `SWT_dual` = `1151.84`
2. `SWT_original` = `1153.88`
3. `FFT_original` = `1525.21`
4. `FFT_dual` = `1859.61`

Relevant EDA signals:

- `MeanAbsCorr = 0.866`, extremely high
- `AutocorrHalfLife = 31`
- `SpectralEntropy = 0.388`, which looks FFT-friendly
- `WaveletDetailPct = 2.32`
- `LocalVarianceRatio = 0.051`

Why `SWT` beats `FFT` even though the spectrum looks favorable:

At first glance, the low spectral entropy suggests FFT should do well. But the EDA also shows this dataset is extremely multivariate, with hundreds of strongly related sensors. In such traffic sensor data, the shared periodicity is not perfectly phase-aligned across nodes. Different sensors may follow similar rhythms with shifted timing and local disruptions. A global spectral representation can lose some of that local structure, while SWT is still local enough to preserve it.

There is also a modeling reason: in this project, FFT is introduced as a tokenization replacement under a fixed backbone. That makes the comparison fair, but it also means FFT does not get a fully redesigned architecture specialized for global frequency processing. So a low-entropy spectrum does not automatically guarantee a win.

Why `dual` beats `original`:

The improvement from `SWT_original` to `SWT_dual` is very small, but it is real. The huge cross-channel dependence means the original branch is already very useful. The temporal branch adds a little more value because the series are still temporally persistent. So `dual` helps, but only slightly, because channel structure is already doing most of the work.

Why the final ranking makes sense:

The big story here is not just "SWT wins." It is that the whole `FFT` side underperforms heavily, while `SWT_original` and `SWT_dual` are almost tied at the top. That matches an EDA picture of a dataset dominated by strong multivariate structure and moderately regular but not globally uniform temporal behavior.

### `Solar`

Ranking by `MSE`:

1. `SWT_original` = `0.1710`
2. `SWT_dual` = `0.1775`
3. `FFT_original` = `0.1892`
4. `FFT_dual` = `0.2287`

Relevant EDA signals:

- `MeanAbsCorr = 0.908`, the highest among the reported datasets
- `AutocorrHalfLife = 18`
- `SpectralEntropy = 0.407`, which is favorable to FFT
- `WaveletDetailPct = 10.97`
- `LocalVarianceRatio = 0.102`
- `ZeroRatio = 0.551` in the EDA overview, which is very high

Why `SWT` beats `FFT`:

`Solar` is the clearest case where looking only at spectral entropy would be misleading. Yes, the data are periodic, and the spectrum is concentrated enough that FFT should be competitive. But solar power also contains hard local transitions: sunrise, sunset, cloud-driven fluctuations, and long zero or near-zero periods at night. Those transitions are abrupt and local. SWT preserves those local changes better than a purely global frequency representation, which explains why `SWT_original` beats both FFT variants.

Why `original` beats `dual`:

The temporal structure here is already simple and strong. Much of it comes from a fairly regular day-night cycle. The original branch already captures enough of that structure under this backbone, so the additional temporal path does not improve things and may even overfit or over-mix the representation.

Why the final ranking makes sense:

`Solar` looks frequency-friendly globally, but local discontinuities still matter in forecasting. That is why `SWT` wins, and the simpler `original` attention path is enough.

### `electricity`

Ranking by `MSE`:

1. `SWT_dual` = `0.1935`
2. `FFT_dual` = `0.1937`
3. `SWT_original` = `0.2021`
4. `FFT_original` = `0.2033`

Relevant EDA signals:

- `MeanAbsCorr = 0.403`
- `AutocorrHalfLife = 9`
- `TemporalProfileStrength = 0.100`
- `SpectralEntropy = 0.652`, the worst for FFT among the reported datasets
- `WaveletDetailPct = 24.19`
- `LocalVarianceRatio = 0.446`

Why `SWT` beats `FFT`:

The spectral entropy is high, which means the frequency content is diffuse rather than concentrated. This is a weak setting for FFT. At the same time, wavelet detail is clearly larger than in the smoother datasets, and local variance is fairly high. That means local multi-scale behavior matters, which naturally favors SWT.

Why `dual` beats `original`:

The gain is small but consistent. `electricity` has enough temporal persistence for a temporal branch to help, but not such a strong clean daily profile that the effect becomes dramatic. So the dual branch improves the representation a little, but the choice of tokenization matters at least as much as the choice of attention fusion.

Why the final ranking makes sense:

This is a near-tie between the two dual models, but the EDA still slightly favors `SWT_dual`. The data are not spectrally simple enough for FFT to dominate, and the moderate local detail keeps SWT just ahead.

### `smartbuilding`

Ranking by `MSE`:

1. `SWT_dual` = `0.5622`
2. `FFT_dual` = `0.5632`
3. `SWT_original` = `0.5691`
4. `FFT_original` = `0.5701`

Relevant EDA signals:

- `MeanAbsCorr = 0.387`
- `AutocorrHalfLife = 3`
- `TemporalProfileStrength = 0.759`, one of the strongest
- `SpectralEntropy = 0.495`
- `WaveletDetailPct = 44.34`, extremely high
- `LocalVarianceRatio = 0.632`, also very high

Why `SWT` beats `FFT`:

This dataset has strong local structure for a physical reason. Loads, HVAC equipment, lighting, plugs, and environmental variables create short-range transitions driven by occupancy and control actions. The smartbuilding plots in the EDA also show clear zone totals, floor total, temperature, and light interactions. This is exactly the kind of multiscale, locally varying behavior SWT is good at preserving.

Why `dual` beats `original`:

Even though the autocorrelation half-life is short, the repeated daily profile is very strong. That means the data contain both local abrupt behavior and repeated time-of-day structure. The original channel/scale mechanism alone is not enough; the temporal branch helps the model use the occupancy-driven and schedule-driven temporal pattern.

Why the final ranking makes sense:

This is the most textbook justification for `SWT_dual`. The EDA says the dataset is simultaneously multivariate, locally variable, and temporally structured. That is exactly the regime where local tokenization plus parallel temporal modeling should win.

### `traffic`

Ranking by `MSE`:

1. `SWT_dual` = `0.5334`
2. `FFT_dual` = `0.5613`
3. `SWT_original` = `0.5777`
4. `FFT_original` = `0.6019`

Relevant EDA signals:

- `MeanAbsCorr = 0.526`
- `AutocorrHalfLife = 2`
- `TemporalProfileStrength = 0.655`
- `SpectralEntropy = 0.601`
- `WaveletDetailPct = 50.35`, the highest in the table
- `LocalVarianceRatio = 0.847`, also the highest

Why `SWT` beats `FFT`:

This is the clearest tokenization win for SWT. The EDA shows that `traffic` has the strongest local variability and the strongest wavelet detail percentage of any reported dataset. Traffic data change abruptly because of congestion spikes, release, rush hour edges, and local incidents. Those are precisely the kinds of short-range patterns that FFT tends to smooth into global frequency content, while SWT preserves them.

Why `dual` beats `original`:

The half-life is very short, so long-range memory is not the main story. But the temporal profile strength is still high, meaning there is a strong repeated day-cycle pattern. The dual branch helps the model capture that repeated temporal rhythm on top of the local multiscale tokenization.

Why the final ranking makes sense:

The winner has a much larger margin here than on most other datasets. That makes sense because both parts of the EDA point in the same direction: strong local detail says "use SWT," and strong daily temporal structure says "add the temporal branch." So `SWT_dual` is exactly the expected winner.

### `weather`

Ranking by `MSE`:

1. `FFT_dual` = `0.1916`
2. `SWT_dual` = `0.1924`
3. `SWT_original` = `0.1929`
4. `FFT_original` = `0.1977`

Relevant EDA signals:

- `MeanAbsCorr = 0.323`
- `AutocorrHalfLife = 72`, the highest by far
- `TemporalProfileStrength = 0.039`, very weak daily shape in the selected series
- `SpectralEntropy = 0.241`, the lowest in the table
- `WaveletDetailPct = 0.000027`, essentially zero
- `LocalVarianceRatio = 0.000953`, also essentially zero

Why `FFT` beats `SWT`:

This is the cleanest FFT-favorable dataset in the results. The EDA says the signal is extremely smooth, highly persistent, and spectrally concentrated, with almost no local detail. That is exactly the regime where a frequency-based tokenization should work well. Since SWT's main advantage is local multiscale detail, weather gives it almost nothing special to exploit.

Why `dual` beats `original`:

The very high autocorrelation half-life means long temporal dependence is the defining feature of the dataset. Even though the chosen representative series does not show a strong hour-of-day profile, the long persistence itself is enough to justify the temporal branch. So `FFT_dual` becomes the best combination: FFT for smooth spectral structure, and dual attention for long temporal dependence.

Why the final ranking makes sense:

If one dataset was expected to break the general SWT trend, this was it. The EDA says weather is the smoothest and most frequency-concentrated case by far, and the results confirm exactly that.

## Why `FFT_original` Never Wins

The most important cross-dataset observation is that `FFT_original` never takes first place. The EDA explains why:

- only `weather` is a very strong clean FFT case
- `Solar` looks FFT-friendly globally, but local sunrise/sunset transitions and zeros still matter
- `PEMS03` has low spectral entropy, but the networked sensor structure is not globally phase-aligned
- `electricity`, `traffic`, and `smartbuilding` have too much local detail or diffuse frequency content
- on datasets where temporal structure matters, `dual` usually improves over `original`

So FFT only wins when both conditions are satisfied:

1. the data are genuinely smooth and spectrally concentrated
2. the temporal branch is present to exploit long dependence

Among these eight datasets, only `weather` fits that combination cleanly.

## Why `SWT_dual` Wins Most Often

`SWT_dual` wins on five of the eight datasets because it is the safest combination for heterogeneous real-world signals:

- SWT preserves local changes and multiscale detail
- the dual branch adds explicit temporal reasoning
- the original channel/scale mechanism is still retained, so multivariate structure is not lost

This is why `SWT_dual` performs especially well on:

- `traffic`
- `smartbuilding`
- `electricity`
- `ETTh1`
- `PEMS03`

These datasets are different, but they share one useful property: the signal is not just smooth and periodic. There is still enough local or multiscale structure that a purely frequency-first representation is not the most robust choice.

## Final Conclusion

The detailed results are consistent with the EDA:

- `weather` is the only clear spectral-smooth case, so `FFT_dual` wins there.
- `traffic` and `smartbuilding` are the clearest local multiscale cases, so `SWT_dual` wins there.
- `electricity` is a mild SWT-favoring case because its spectrum is diffuse and its local detail is moderate.
- `ETTh1` is balanced but still slightly favors `SWT_dual` because repeated temporal structure and local variation both matter.
- `ETTm1` and `Solar` prefer the simpler `SWT_original` because their useful structure is already captured reasonably well without the extra temporal fusion.
- `PEMS03` is dominated by strong multivariate coupling, so `SWT` stays strong and `dual` gives only a small additional gain.

In short:

SWT wins when locality and multi-scale detail matter, FFT wins when the signal is smooth and frequency-concentrated, and the dual branch helps when there is additional temporal structure that the original channel/scale mechanism alone does not fully exploit.

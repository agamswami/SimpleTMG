# Reasons With EDA

This note explains the results in [final_metrics.csv](/f:/files/work/btp/simpletm/simpleTMG/final_metrics.csv) using the saved exploratory analysis in [eda-analysis.ipynb](/f:/files/work/btp/simpletm/simpleTMG/eda-analysis.ipynb).

The comparison should be read dataset-by-dataset. The datasets are on very different scales, so a lower MSE on one dataset does not mean that dataset is "easier" than another. What matters is which variant wins inside the same dataset.

## How To Read The Result

- Lower is better for `MSE`, `MAE`, `RMSE`, `MAPE`, `SMAPE`, and `WAPE`.
- Higher is better for `R2`.
- The EDA indicators used here are:
  - `MeanAbsCorr`: strength of cross-channel dependence
  - `AutocorrHalfLife` and `TemporalProfileStrength`: temporal persistence and repeated daily structure
  - `SpectralEntropy`: lower means a cleaner and more concentrated frequency structure, which is more FFT-friendly
  - `WaveletDetailPct`: higher means more multi-scale local detail, which is more SWT-friendly
  - `LocalVarianceRatio`: higher means stronger local short-range fluctuations

## Main Pattern

The final results show:

- `SWT_dual` is best on `ETTh1`, `PEMS03`, `electricity`, `smartbuilding`, and `traffic`.
- `SWT_original` is best on `ETTm1` and `Solar`.
- `FFT_dual` is best on `weather`.
- `FFT_original` does not win on any of the reported datasets.

This matches the overall EDA story reasonably well:

- Datasets with strong local variation and high wavelet detail tend to favor `SWT`.
- Datasets with very clean long-range periodic structure and almost no local detail are the best place for `FFT`.
- The dual-attention branch helps most when temporal structure exists in addition to channel coupling.

## Cross-Dataset Interpretation

The original SimpleTM channel/scale mechanism is justified almost everywhere because cross-channel dependence is not small on any reported dataset. `Solar`, `PEMS03`, `traffic`, `electricity`, and `smartbuilding` all have substantial cross-channel correlation, so treating each variable independently would discard useful structure.

The temporal branch is not equally useful everywhere. It helps most when the data show either long persistence or a strong repeated hour-of-day profile. That is why the dual variant improves on `ETTh1`, `weather`, `traffic`, `smartbuilding`, and `electricity`, but is neutral or slightly harmful on `ETTm1` and `Solar`.

The tokenization comparison is also consistent with the EDA. `SWT` keeps locality and multi-scale detail, so it works well on datasets with bursts, regime changes, or strong local fluctuations. `FFT` is most useful when the spectrum is concentrated and the local detail is weak. Among the reported datasets, `weather` is the cleanest example of that case.

## Dataset-Wise Reasons

| Dataset | Best Variant | Key EDA Signals | Why the result makes sense |
| --- | --- | --- | --- |
| `ETTh1` | `SWT_dual` | Medium channel correlation (`0.222`), low half-life (`6`), strong daily profile (`0.661`), moderate spectral entropy (`0.584`), some wavelet detail (`8.08`) | This dataset has meaningful daily structure but also non-trivial local variation. The temporal branch helps a little, and `SWT` stays slightly ahead of `FFT` because preserving local structure matters more than a purely frequency-based tokenization. |
| `ETTm1` | `SWT_original` | Medium channel correlation (`0.224`), long half-life (`29`), strong daily profile (`0.658`), lower spectral entropy (`0.464`), very low wavelet detail (`1.20`) | The data are temporally persistent and smoother than `ETTh1`. That reduces the need for extra fusion complexity. The original attention is already enough, so the dual branch does not help here. The `SWT` vs `FFT` gap is small, which matches the EDA: this dataset is not strongly local-detail dominated. |
| `PEMS03` | `SWT_dual` | Very high channel correlation (`0.866`), long half-life (`31`), low spectral entropy (`0.388`), low wavelet detail (`2.32`) | EDA says `PEMS03` is highly structured and very multivariate. The strong win of the dual model suggests that adding temporal reasoning helps on top of the heavy channel coupling. Although the spectrum is FFT-friendly, the fixed SimpleTM backbone still works better when tokenization preserves local structure, so `SWT` remains ahead in this implementation. |
| `Solar` | `SWT_original` | Very high channel correlation (`0.908`), medium half-life (`18`), low spectral entropy (`0.407`), moderate wavelet detail (`10.97`), very high zero ratio in the EDA overview | `Solar` is one of the most FFT-friendly datasets in the EDA, but it also contains sharp day/night transitions and many zeros. Those local on/off transitions are exactly the type of structure `SWT` preserves better. That explains why `FFT` is competitive in principle but still loses here. |
| `electricity` | `SWT_dual` | High channel correlation (`0.403`), medium half-life (`9`), high spectral entropy (`0.652`), medium wavelet detail (`24.19`), fairly high local variance (`0.446`) | The EDA does not favor `FFT` here because the spectrum is relatively diffuse. At the same time, there is enough local variation and multi-scale behavior to make `SWT` useful, and the temporal branch gives a small gain because the load still has temporal persistence. |
| `smartbuilding` | `SWT_dual` | High channel correlation (`0.387`), short half-life (`3`), very strong daily profile (`0.759`), high wavelet detail (`44.34`), high local variance (`0.632`) | This is the clearest `SWT + dual attention` case. The smartbuilding data mix equipment loads, zone totals, temperature, humidity, and lux. The EDA shows both strong multivariate coupling and strong local multi-scale variation. The temporal branch helps because occupancy-driven daily patterns are strong, while `SWT` helps because the local transitions are sharp. |
| `traffic` | `SWT_dual` | High channel correlation (`0.526`), very short half-life (`2`), strong daily profile (`0.655`), high wavelet detail (`50.35`), highest local variance (`0.847`) | `traffic` is dominated by local fluctuations and short-range changes. This is exactly where `SWT` should help most, and the results confirm it. The dual temporal branch also helps because, despite the short half-life, the day-cycle profile is still strong. `FFT` is weaker because a global frequency view loses too much of the sharp local behavior. |
| `weather` | `FFT_dual` | High channel correlation (`0.323`), longest half-life (`72`), extremely low spectral entropy (`0.241`), almost zero wavelet detail (`0.000027`), almost zero local variance (`0.000953`) | This is the cleanest FFT win in the table. The EDA says the signal is smooth, highly persistent, and dominated by a compact frequency structure, with almost no transient multi-scale detail. That is exactly the condition where `FFT` should perform well. The dual branch then adds value because the time dependence is very long-range. |

## Why Dual Attention Helps Sometimes But Not Always

The dual branch is most useful when the dataset has both:

- non-trivial channel coupling
- usable temporal structure that the original branch does not fully capture

That is why it helps on `ETTh1`, `PEMS03`, `electricity`, `smartbuilding`, `traffic`, and `weather`.

It helps less on `ETTm1` and `Solar` because those datasets are already handled reasonably well by the original branch under the fixed SimpleTM backbone. In those cases, the added temporal path does not create enough extra information to offset the added fusion complexity.

## Why SWT Usually Beats FFT In These Runs

Across these reported runs, `SWT` wins more often because several datasets in the result table still contain meaningful local structure:

- `traffic` and `smartbuilding` have very high `WaveletDetailPct`
- `electricity` has moderate wavelet detail and diffuse frequency content
- `ETTh1` has more local variation than `ETTm1`
- `Solar` has sharp day/night switching despite looking frequency-friendly globally

So even when some datasets look FFT-friendly in the spectrum, the current fixed-backbone comparison still benefits from keeping temporal locality in the tokenization.

## Short Conclusion

The EDA explains the results well:

- `SWT` wins when local, multi-scale, and bursty behavior matters.
- `FFT` wins when the signal is smooth, highly persistent, and spectrally concentrated.
- The dual-attention branch helps when temporal structure exists in addition to channel coupling.
- `smartbuilding` is a valuable added dataset because it stresses all three aspects at once: multivariate coupling, strong daily behavior, and strong local multi-scale changes.

If this is turned into report text, the cleanest one-line summary is:

`The observed ranking is consistent with the EDA: SWT-based tokenization is stronger on datasets with higher local detail and non-stationary multi-scale behavior, whereas FFT becomes competitive or best only when the data are smoother, more periodic, and spectrally concentrated, with weather being the clearest example.`

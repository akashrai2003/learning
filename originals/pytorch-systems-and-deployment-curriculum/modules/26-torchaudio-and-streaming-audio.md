# Module 26 — TorchAudio, TorchCodec, and Streaming Audio

**Phase:** 4 — PyTorch model engineering and optimization  
**Time:** 9 focused hours  
**Prerequisites:** PyTorch data pipelines, signal-processing basics, and Module 25’s contract mindset.

## Existing inference curriculum bridge

Relate this to [03 — Tensor math and shapes](../../inference-engineering-curriculum/modules/03-tensor-math-shapes-and-einsum.md), [15 — Multimodal inference](../../inference-engineering-curriculum/modules/15-multimodal-inference.md), [16 — SSM, recurrent, and hybrid models](../../inference-engineering-curriculum/modules/16-ssm-recurrent-and-hybrid-models.md), [34 — Continuous batching/chunked prefill](../../inference-engineering-curriculum/modules/34-continuous-batching-chunked-prefill-and-prefix-cache.md), and [44 — Profiling and correctness](../../inference-engineering-curriculum/modules/44-profiling-benchmarking-and-correctness.md).

## Why this matters and outcomes

Audio applications force you to confront time, buffering, resampling, and latency budgets. They are an unusually good way to learn that “streaming” is an end-to-end protocol, not a model flag. They also illustrate API lifecycle engineering: starting in TorchAudio 2.8, much of its I/O moved to maintenance/deprecation and decoding/encoding is being consolidated into TorchCodec. Learn TorchAudio’s signal/model pieces, but make new I/O code conscious of the pinned release and migration path.

You will implement chunk-safe feature extraction, account for algorithmic latency, distinguish offline and online model state, profile real-time factor, and design an audio pipeline whose I/O dependency can migrate without changing model semantics.

## Guided study sources

Follow this sequence alongside the module’s implementation ladder. The full
four-source progression, including the production source to inspect, is in the
[PyTorch systems source atlas](../resources/SOURCE_ATLAS.md) under **P26 — Torchaudio and streaming audio**.

1. [TorchAudio documentation](https://docs.pytorch.org/audio/stable/) — learn the API or system contract.
2. [TorchCodec documentation](https://github.com/pytorch/torchcodec) — study the underlying design, paper, or specification.
3. [TorchAudio repository](https://github.com/pytorch/audio) — implement, profile, or validate the behavior.

Use the existing source section below as supplemental reading; the ordered path
above tells you what to learn and build first.

### Further study and source walks

- [TorchAO source](https://github.com/pytorch/ao) — read this after the core sequence; extract one idea, code path, or experiment for your notes.
- [Torchtune source](https://github.com/pytorch/torchtune) — read this after the core sequence; extract one idea, code path, or experiment for your notes.
- [PyTorch AMP docs](https://docs.pytorch.org/docs/stable/amp.html) — read this after the core sequence; extract one idea, code path, or experiment for your notes.

## Mental model and theory

A waveform is commonly `(C,T)` or batched `(B,C,T)`, at sampling rate \(f_s\). A frame of \(N\) samples with hop \(H\) creates a time-frequency representation with frame rate \(f_s/H\). The STFT is

\[
X[m,k]=\sum_{n=0}^{N-1}x[n+mH]w[n]e^{-j2\pi kn/N}.
\]

Mel features apply a nonnegative filterbank to \(|X|^2\), then often log compression. Window, center padding, hop, FFT size, mel scale, dB/log floor, channel mixing, and input sample rate all form the model contract. “16 kHz mono” is insufficient when a deployment uses a distinct resampler or frame boundary convention.

Streaming introduces a left context/state plus new chunk. If a model needs \(L\) samples of lookahead, chunk duration is \(C/f_s\), and compute duration is \(t_c\), a rough first-output latency is capture/buffer delay plus \(L/f_s+t_c\). Real-time factor is processing seconds divided by audio seconds; it must stay below one in sustained operation. Chunking an offline convolution/attention model without preserving receptive-field state creates boundary artifacts. RNN-T/Emformer-like systems have explicit cache/state contracts; test chunk concatenation against an offline oracle where the architecture permits it.

Use TorchCodec as the forward-looking decode/encode boundary and keep tensors entering the feature extractor canonical. If you study historical `torchaudio.io.StreamReader`, label it deprecated in your notes rather than building production dependence on it. TorchAudio’s functional transforms, datasets, and certain model/decoder components remain useful, but check version-specific availability before selecting a lab.

## Implementation ladder in `torch-lab/`

1. **Reference:** create `torch-lab/audio/reference_features.py` with a pure-Torch waveform generator, STFT/mel/log pipeline, shape assertions, and a frame-by-frame offline oracle.
2. **Optimized:** add `torch-lab/audio/streaming_pipeline.py` using a pinned TorchCodec decoder plus TorchAudio/PyTorch transforms where available. Process fixed chunks, maintain explicit state, and batch compatible streams only after measuring latency.
3. **Integrated:** connect an audio encoder to a small language decoder or classifier; make a real-time dashboard showing decode, resample, feature, H2D, encoder, decoder, and queueing time. Export a supported static path only after offline/streaming parity passes.

## Three focused sessions

1. **DSP contract:** derive and implement STFT/mel dimensions, sweep sample rate/hop/window, and compare a known waveform’s features across chunk boundaries.
2. **Online state:** implement overlap/state handling, inject jitter and discontinuities, and quantify first-token latency, chunk latency, RTF, and boundary error.
3. **Lifecycle/source walk:** inspect pinned TorchAudio/TorchCodec interfaces and write a migration note from deprecated I/O to the target decoder API.

## Experiments and deliverables

Commit `audio_contract.yaml`, `stream_parity_test.py`, `latency_budget.csv`, and `io_migration.md`. Include codec/container, decoder version, source and target rates, channel/downmix policy, feature parameters, chunk/lookahead, state serialization, utterance boundaries, VAD policy, RTF, p50/p99 latency, and transcripts/class outputs.

## Common failure modes

- Measuring model compute while excluding buffering and decoding latency.
- Resampling each chunk independently and creating discontinuities.
- Mixing `(T,C)` and `(C,T)` conventions silently.
- Using deprecated TorchAudio I/O in new code without a pinned migration strategy.
- Claiming streaming parity where a bidirectional/offline model necessarily has future context.

## Mastery gate

Produce a streaming pipeline that processes a five-minute clip with bounded memory and documented RTF. Explain every state tensor and latency term, demonstrate acceptable offline/online parity on a fixed test set, and show how the decoder layer can be replaced without changing downstream tensor semantics.

## Primary sources, code paths, and OSS bridge

- [TorchAudio stable documentation and maintenance notice](https://docs.pytorch.org/audio/stable/)
- [TorchCodec source and installation guidance](https://github.com/meta-pytorch/torchcodec)
- [TorchAudio streaming tutorial (historical/deprecation context)](https://github.com/pytorch/audio)
- [TorchAudio repository](https://github.com/pytorch/audio)
- [TorchAudio paper](https://arxiv.org/abs/2110.15018)

Read `src/torchaudio/functional/`, `src/torchaudio/models/`, `src/torchaudio/models/decoder/`, and the corresponding TorchCodec decoder code/tests for your pinned release. Contributions worth pursuing are feature/chunk parity tests, a migration reproducer, decoder error messages, or a minimal model-export issue—not speculative kernel changes.

## Next

Module 27 composes vision, audio, and text into an explicit multimodal token and preprocessing contract.

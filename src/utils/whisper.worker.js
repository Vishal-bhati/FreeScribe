import { pipeline } from '@xenova/transformers'
import { MessageTypes } from './presets'

class MyTranscriptionPipeline {
    static task = 'automatic-speech-recognition'
    static model = 'openai/whisper-tiny.en'
    static instance = null

    static async getInstance(progressCallback = null) {
        if (this.instance === null) {
            try {
                this.instance = await pipeline(this.task, null, { progressCallback })
            } catch (error) {
                console.error("Failed to create pipeline instance:", error.message)
            }
        }
        return this.instance
    }
}

self.addEventListener('message', async (event) => {
    const { type, audio } = event.data
    if (type === MessageTypes.INFERENCE_REQUEST) {
        await transcribe(audio)
    }
})

async function transcribe(audio) {
    sendLoadingMessage('loading')

    let pipeline
    try {
        pipeline = await MyTranscriptionPipeline.getInstance(loadModelCallback)
    } catch (error) {
        console.error("Pipeline creation error:", error.message)
        sendLoadingMessage('error')
        return
    }

    sendLoadingMessage('success')

    const strideLengthS = 5
    const generationTracker = new GenerationTracker(pipeline, strideLengthS)

    try {
        await pipeline(audio, {
            top_k: 0,
            do_sample: false,
            chunk_length: 30,
            stride_length_s: strideLengthS,
            return_timestamps: true,
            callback_function: generationTracker.callbackFunction.bind(generationTracker),
            chunk_callback: generationTracker.chunkCallback.bind(generationTracker)
        })
    } catch (error) {
        console.error("Transcription error:", error.message)
    }
    generationTracker.sendFinalResult()
}

async function loadModelCallback(data) {
    const { status } = data
    if (status === 'progress') {
        const { file, progress, loaded, total } = data
        sendDownloadingMessage(file, progress, loaded, total)
    }
}

function sendLoadingMessage(status) {
    self.postMessage({
        type: MessageTypes.LOADING,
        status
    })
}

function sendDownloadingMessage(file, progress, loaded, total) {
    self.postMessage({
        type: MessageTypes.DOWNLOADING,
        file,
        progress,
        loaded,
        total
    })
}

class GenerationTracker {
    constructor(pipeline, strideLengthS) {
        this.pipeline = pipeline
        this.strideLengthS = strideLengthS
        this.chunks = []
        this.timePrecision = pipeline?.processor.feature_extractor.config.chunk_length / pipeline.model.config.max_source_positions
        this.processedChunks = []
        this.callbackFunctionCounter = 0
    }

    sendFinalResult() {
        self.postMessage({ type: MessageTypes.INFERENCE_DONE })
    }

    callbackFunction(beams) {
        this.callbackFunctionCounter += 1
        if (this.callbackFunctionCounter % 10 !== 0) {
            return
        }

        const bestBeam = beams[0]
        const text = this.pipeline.tokenizer.decode(bestBeam.output_token_ids, {
            skip_special_tokens: true
        })

        const result = {
            text,
            start: this.getLastChunkTimestamp(),
            end: undefined
        }

        createPartialResultMessage(result)
    }

    chunkCallback(data) {
        this.chunks.push(data)
        const [text, { chunks }] = this.pipeline.tokenizer._decode_asr(
            this.chunks,
            {
                time_precision: this.timePrecision,
                return_timestamps: true,
                force_full_sequence: false
            }
        )

        this.processedChunks = chunks.map((chunk, index) => this.processChunk(chunk, index))

        createResultMessage(
            this.processedChunks, false, this.getLastChunkTimestamp()
        )
    }

    getLastChunkTimestamp() {
        if (this.processedChunks.length === 0) {
            return 0
        }
        return this.processedChunks[this.processedChunks.length - 1].end || 0
    }

    processChunk(chunk, index) {
        const { text, timestamp } = chunk
        const [start, end] = timestamp

        return {
            index,
            text: text.trim(),
            start: Math.round(start),
            end: Math.round(end) || Math.round(start + 0.9 * this.strideLengthS)
        }
    }
}

function createResultMessage(results, isDone, completedUntilTimestamp) {
    self.postMessage({
        type: MessageTypes.RESULT,
        results,
        isDone,
        completedUntilTimestamp
    })
}

function createPartialResultMessage(result) {
    self.postMessage({
        type: MessageTypes.RESULT_PARTIAL,
        result
    })
}

/**
 * AI-POWERED DEEPFAKE VOICE DETECTION
 * Live Microphone Recording & Audio Analysis Controller (recorder.js)
 * 
 * Uses Web Audio API & MediaStream to capture genuine microphone audio,
 * visualize the live waveform in real-time, encode PCM 16-bit WAV on the client,
 * and dispatch to the Flask CNN backend.
 */

class LiveVoiceRecorder {
    constructor() {
        this.audioContext = null;
        this.mediaStream = null;
        this.sourceNode = null;
        this.analyserNode = null;
        this.processorNode = null;

        this.audioData = [];
        this.recordingLength = 0;
        this.sampleRate = 44100;

        this.isRecording = false;
        this.isPaused = false;
        this.recordedBlob = null;

        this.timerInterval = null;
        this.startTime = null;
        this.elapsedSeconds = 0;
        this.animationFrameId = null;

        this.initDOMElements();
        this.bindEvents();
    }

    initDOMElements() {
        this.btnStart = document.getElementById('btnStartRecord');
        this.btnStop = document.getElementById('btnStopRecord');
        this.btnPause = document.getElementById('btnPauseRecord');
        this.btnResume = document.getElementById('btnResumeRecord');
        this.btnRerecord = document.getElementById('btnRerecord');
        this.btnAnalyze = document.getElementById('btnAnalyzeRecord');
        this.btnDelete = document.getElementById('btnDeleteRecord');

        this.timerDisplay = document.getElementById('recordingTimer');
        this.statusBadge = document.getElementById('micStatusBadge');
        this.statusText = document.getElementById('micStatusText');
        this.playbackSection = document.getElementById('playbackSection');
        this.recordedAudioPlayer = document.getElementById('recordedAudioPlayer');
        this.recordedDurationText = document.getElementById('recordedDurationText');
        this.errorAlert = document.getElementById('micErrorAlert');
        this.errorMessage = document.getElementById('micErrorMessage');
        this.visualizerCanvas = document.getElementById('micVisualizer');
        this.loadingOverlay = document.getElementById('loadingOverlay');

        if (this.visualizerCanvas) {
            this.canvasCtx = this.visualizerCanvas.getContext('2d');
            this.drawIdleVisualizer();
        }
    }

    bindEvents() {
        if (this.btnStart) this.btnStart.addEventListener('click', () => this.startRecording());
        if (this.btnStop) this.btnStop.addEventListener('click', () => this.stopRecording());
        if (this.btnPause) this.btnPause.addEventListener('click', () => this.pauseRecording());
        if (this.btnResume) this.btnResume.addEventListener('click', () => this.resumeRecording());
        if (this.btnRerecord) this.btnRerecord.addEventListener('click', () => this.resetRecorder());
        if (this.btnDelete) this.btnDelete.addEventListener('click', () => this.resetRecorder());
        if (this.btnAnalyze) this.btnAnalyze.addEventListener('click', () => this.analyzeRecording());
    }

    showError(msg) {
        if (this.errorAlert && this.errorMessage) {
            this.errorMessage.textContent = msg;
            this.errorAlert.style.display = 'block';
        } else {
            alert(msg);
        }
    }

    clearError() {
        if (this.errorAlert) this.errorAlert.style.display = 'none';
    }

    setStatus(state, text) {
        if (!this.statusBadge || !this.statusText) return;
        this.statusBadge.className = 'status-indicator status-' + state;
        this.statusText.textContent = text;
    }

    async startRecording() {
        this.clearError();

        // 1. Check browser support
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            this.showError("Microphone access is not supported by your browser or requires HTTPS / localhost.");
            return;
        }

        try {
            // 2. Request microphone permission on user interaction
            this.mediaStream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    channelCount: 1,
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true
                },
                video: false
            });

            // 3. Initialize AudioContext at native hardware sample rate
            const AudioContextClass = window.AudioContext || window.webkitAudioContext;
            this.audioContext = new AudioContextClass();
            this.sampleRate = this.audioContext.sampleRate;

            this.sourceNode = this.audioContext.createMediaStreamSource(this.mediaStream);
            this.analyserNode = this.audioContext.createAnalyser();
            this.analyserNode.fftSize = 2048;

            // ScriptProcessorNode for pristine raw PCM buffer capture
            const bufferSize = 4096;
            this.processorNode = this.audioContext.createScriptProcessor(bufferSize, 1, 1);

            this.audioData = [];
            this.recordingLength = 0;

            this.processorNode.onaudioprocess = (e) => {
                if (!this.isRecording || this.isPaused) return;
                const channelData = e.inputBuffer.getChannelData(0);
                this.audioData.push(new Float32Array(channelData));
                this.recordingLength += channelData.length;
            };

            this.sourceNode.connect(this.analyserNode);
            this.analyserNode.connect(this.processorNode);
            this.processorNode.connect(this.audioContext.destination);

            // 4. Update UI State
            this.isRecording = true;
            this.isPaused = false;
            this.elapsedSeconds = 0;
            this.startTime = Date.now();

            this.btnStart.style.display = 'none';
            this.btnStop.style.display = 'inline-flex';
            this.btnPause.style.display = 'inline-flex';
            this.btnResume.style.display = 'none';
            if (this.playbackSection) this.playbackSection.style.display = 'none';

            this.setStatus('recording', 'Recording Live Speech...');
            this.startTimer();
            this.drawActiveVisualizer();

        } catch (err) {
            console.error("Microphone capture error:", err);
            if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
                this.showError("Microphone permission was denied. Please allow microphone access in your browser settings to perform live voice analysis.");
            } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
                this.showError("No microphone hardware was detected on this device. Please connect a microphone and try again.");
            } else {
                this.showError("Failed to access microphone: " + err.message);
            }
            this.setStatus('idle', 'Microphone Error');
        }
    }

    pauseRecording() {
        if (!this.isRecording || this.isPaused) return;
        this.isPaused = true;
        this.btnPause.style.display = 'none';
        this.btnResume.style.display = 'inline-flex';
        this.setStatus('paused', 'Recording Paused');
        clearInterval(this.timerInterval);
    }

    resumeRecording() {
        if (!this.isRecording || !this.isPaused) return;
        this.isPaused = false;
        this.btnPause.style.display = 'inline-flex';
        this.btnResume.style.display = 'none';
        this.setStatus('recording', 'Recording Live Speech...');
        this.startTimer();
    }

    stopRecording() {
        if (!this.isRecording) return;
        this.isRecording = false;
        this.isPaused = false;
        clearInterval(this.timerInterval);

        if (this.animationFrameId) {
            cancelAnimationFrame(this.animationFrameId);
        }

        // Cleanly terminate audio nodes and hardware microphone tracks
        if (this.processorNode) {
            this.processorNode.disconnect();
            this.processorNode = null;
        }
        if (this.sourceNode) {
            this.sourceNode.disconnect();
            this.sourceNode = null;
        }
        if (this.mediaStream) {
            this.mediaStream.getTracks().forEach(track => track.stop());
            this.mediaStream = null;
        }

        // Validate minimum duration (at least 1.0 second)
        const totalDurationSec = this.recordingLength / this.sampleRate;
        if (totalDurationSec < 1.0) {
            this.showError("Recording is too short (" + totalDurationSec.toFixed(1) + "s). Please record at least 2.0 - 4.0 seconds of natural speech for reliable deepfake detection.");
            this.resetRecorder();
            return;
        }

        // Encode recorded raw PCM Float32 buffers into standard 16-bit PCM WAV at native sample rate
        // Backend Librosa will perform high-fidelity polyphase anti-aliasing resampling to 16 kHz
        this.recordedBlob = this.encodeWAV(this.audioData, this.recordingLength, this.sampleRate);
        const audioUrl = URL.createObjectURL(this.recordedBlob);

        // Update UI
        if (this.recordedAudioPlayer) {
            this.recordedAudioPlayer.src = audioUrl;
        }
        if (this.recordedDurationText) {
            this.recordedDurationText.textContent = totalDurationSec.toFixed(2) + " seconds";
        }
        if (this.playbackSection) {
            this.playbackSection.style.display = 'block';
        }

        this.btnStart.style.display = 'none';
        this.btnStop.style.display = 'none';
        this.btnPause.style.display = 'none';
        this.btnResume.style.display = 'none';

        this.setStatus('idle', 'Recording Completed');
        this.drawIdleVisualizer();
    }

    resetRecorder() {
        this.stopRecording();
        this.recordedBlob = null;
        this.elapsedSeconds = 0;
        if (this.timerDisplay) this.timerDisplay.textContent = '00:00';
        if (this.recordedAudioPlayer) this.recordedAudioPlayer.src = '';
        if (this.playbackSection) this.playbackSection.style.display = 'none';

        this.btnStart.style.display = 'inline-flex';
        this.btnStop.style.display = 'none';
        this.btnPause.style.display = 'none';
        this.btnResume.style.display = 'none';

        this.setStatus('idle', 'Ready to Record');
        this.clearError();
        this.drawIdleVisualizer();
    }

    startTimer() {
        this.timerInterval = setInterval(() => {
            this.elapsedSeconds += 0.5;
            const minutes = Math.floor(this.elapsedSeconds / 60);
            const seconds = Math.floor(this.elapsedSeconds % 60);
            if (this.timerDisplay) {
                this.timerDisplay.textContent = 
                    String(minutes).padStart(2, '0') + ':' + String(seconds).padStart(2, '0');
            }
            // Auto stop at 8 seconds
            if (this.elapsedSeconds >= 8) {
                this.stopRecording();
            }
        }, 500);
    }

    /**
     * Real-time Canvas Waveform Animation
     */
    drawActiveVisualizer() {
        if (!this.analyserNode || !this.canvasCtx || !this.visualizerCanvas) return;

        const canvas = this.visualizerCanvas;
        const ctx = this.canvasCtx;
        const bufferLength = this.analyserNode.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);

        const render = () => {
            if (!this.isRecording) return;
            this.animationFrameId = requestAnimationFrame(render);

            this.analyserNode.getByteTimeDomainData(dataArray);

            ctx.fillStyle = '#070a13';
            ctx.fillRect(0, 0, canvas.width, canvas.height);

            ctx.lineWidth = 2.5;
            // Electric cyan to purple gradient
            const gradient = ctx.createLinearGradient(0, 0, canvas.width, 0);
            gradient.addColorStop(0, '#38bdf8');
            gradient.addColorStop(0.5, '#a855f7');
            gradient.addColorStop(1, '#06b6d4');
            ctx.strokeStyle = gradient;

            ctx.beginPath();
            const sliceWidth = canvas.width * 1.0 / bufferLength;
            let x = 0;

            for (let i = 0; i < bufferLength; i++) {
                const v = dataArray[i] / 128.0;
                const y = v * (canvas.height / 2);

                if (i === 0) {
                    ctx.moveTo(x, y);
                } else {
                    ctx.lineTo(x, y);
                }
                x += sliceWidth;
            }

            ctx.lineTo(canvas.width, canvas.height / 2);
            ctx.stroke();
        };

        render();
    }

    drawIdleVisualizer() {
        if (!this.canvasCtx || !this.visualizerCanvas) return;
        const canvas = this.visualizerCanvas;
        const ctx = this.canvasCtx;

        ctx.fillStyle = '#070a13';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        // Subtle idle horizontal centerline
        ctx.strokeStyle = '#1e293b';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(0, canvas.height / 2);
        ctx.lineTo(canvas.width, canvas.height / 2);
        ctx.stroke();
    }

    /**
     * Pure JavaScript PCM 16-bit RIFF WAV Encoder
     * Preserves native hardware sample rate without crude downsampling.
     */
    encodeWAV(buffers, totalSamples, sampleRate) {
        const mergedBuffer = new Float32Array(totalSamples);
        let offset = 0;
        for (let i = 0; i < buffers.length; i++) {
            mergedBuffer.set(buffers[i], offset);
            offset += buffers[i].length;
        }

        // Create standard RIFF WAV bytes
        const wavBuffer = new ArrayBuffer(44 + mergedBuffer.length * 2);
        const view = new DataView(wavBuffer);

        /* RIFF identifier */
        this.writeString(view, 0, 'RIFF');
        /* file length */
        view.setUint32(4, 36 + mergedBuffer.length * 2, true);
        /* RIFF type */
        this.writeString(view, 8, 'WAVE');
        /* format chunk identifier */
        this.writeString(view, 12, 'fmt ');
        /* format chunk length */
        view.setUint32(16, 16, true);
        /* sample format (raw PCM = 1) */
        view.setUint16(20, 1, true);
        /* channel count (1 = mono) */
        view.setUint16(22, 1, true);
        /* sample rate */
        view.setUint32(24, sampleRate, true);
        /* byte rate (sampleRate * 2 bytes * 1 channel) */
        view.setUint32(28, sampleRate * 2, true);
        /* block align (2 bytes) */
        view.setUint16(32, 2, true);
        /* bits per sample (16 bit) */
        view.setUint16(34, 16, true);
        /* data chunk identifier */
        this.writeString(view, 36, 'data');
        /* data chunk length */
        view.setUint32(40, mergedBuffer.length * 2, true);

        // Write PCM 16-bit audio samples
        let writeOffset = 44;
        for (let i = 0; i < mergedBuffer.length; i++) {
            let s = Math.max(-1, Math.min(1, mergedBuffer[i]));
            view.setInt16(writeOffset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
            writeOffset += 2;
        }

        return new Blob([view], { type: 'audio/wav' });
    }

    writeString(view, offset, string) {
        for (let i = 0; i < string.length; i++) {
            view.setUint8(offset + i, string.charCodeAt(i));
        }
    }

    async analyzeRecording() {
        if (!this.recordedBlob) {
            this.showError("No recorded voice audio to analyze. Please record your voice first.");
            return;
        }

        const formData = new FormData();
        formData.append('audio_file', this.recordedBlob, 'live_microphone_recording.wav');

        if (this.loadingOverlay) this.loadingOverlay.style.display = 'flex';

        try {
            const response = await fetch('/predict-live', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (response.ok && data.status === 'success') {
                sessionStorage.setItem('current_prediction', JSON.stringify(data));
                window.location.href = '/result';
            } else {
                if (this.loadingOverlay) this.loadingOverlay.style.display = 'none';
                this.showError('Analysis Failed: ' + (data.message || 'Error executing Deep Learning model.'));
            }
        } catch (err) {
            if (this.loadingOverlay) this.loadingOverlay.style.display = 'none';
            console.error('Inference error:', err);
            this.showError('Communication error: ' + err.message);
        }
    }
}

document.addEventListener('DOMContentLoaded', () => {
    if (document.getElementById('micVisualizer')) {
        window.voiceRecorder = new LiveVoiceRecorder();
    }
});

/**
 * AI-POWERED DEEPFAKE VOICE DETECTION
 * Client-Side Upload & Dashboard Controller (script.js)
 */

document.addEventListener('DOMContentLoaded', () => {
    initUploadZone();
});

function initUploadZone() {
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('audioFileInput');
    const filePreview = document.getElementById('filePreview');
    const dropPrompt = document.getElementById('dropPrompt');
    const audioPlayer = document.getElementById('audioPlayer');
    const fileNameSpan = document.getElementById('fileName');
    const fileSizeSpan = document.getElementById('fileSize');
    const fileDurationSpan = document.getElementById('fileDuration');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const removeBtn = document.getElementById('removeFileBtn');
    const loadingOverlay = document.getElementById('loadingOverlay');

    if (!dropzone || !fileInput) return;

    let selectedFile = null;

    // Drag-and-drop event listeners
    ['dragenter', 'dragover'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.add('drag-over');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.remove('drag-over');
        }, false);
    });

    dropzone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            handleSelectedFile(files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (fileInput.files.length > 0) {
            handleSelectedFile(fileInput.files[0]);
        }
    });

    function handleSelectedFile(file) {
        // Validate extension
        const validExtensions = ['.wav', '.mp3', '.flac', '.ogg'];
        const nameLower = file.name.toLowerCase();
        const hasValidExt = validExtensions.some(ext => nameLower.endsWith(ext));

        if (!hasValidExt) {
            alert('Unsupported audio format. Please upload a WAV, MP3, FLAC, or OGG file.');
            return;
        }

        // Validate size (max 25MB)
        if (file.size > 25 * 1024 * 1024) {
            alert('File exceeds 25 MB limit. Please select a shorter speech recording.');
            return;
        }

        selectedFile = file;

        // Update preview UI
        fileNameSpan.textContent = file.name;
        fileSizeSpan.textContent = formatBytes(file.size);

        const objectUrl = URL.createObjectURL(file);
        audioPlayer.src = objectUrl;

        audioPlayer.onloadedmetadata = () => {
            const dur = audioPlayer.duration;
            fileDurationSpan.textContent = dur ? dur.toFixed(2) + ' sec' : 'Unknown';
        };

        dropPrompt.style.display = 'none';
        filePreview.style.display = 'block';
        if (analyzeBtn) analyzeBtn.disabled = false;
    }

    if (removeBtn) {
        removeBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            selectedFile = null;
            fileInput.value = '';
            audioPlayer.src = '';
            dropPrompt.style.display = 'block';
            filePreview.style.display = 'none';
            if (analyzeBtn) analyzeBtn.disabled = true;
        });
    }

    // Submit for Deep Learning analysis
    if (analyzeBtn) {
        analyzeBtn.addEventListener('click', async () => {
            if (!selectedFile) {
                alert('Please upload or select an audio file first.');
                return;
            }

            const formData = new FormData();
            formData.append('audio_file', selectedFile);

            if (loadingOverlay) loadingOverlay.style.display = 'flex';

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();

                if (response.ok && data.status === 'success') {
                    // Store prediction result in sessionStorage and navigate to /result
                    sessionStorage.setItem('current_prediction', JSON.stringify(data));
                    window.location.href = '/result';
                } else {
                    if (loadingOverlay) loadingOverlay.style.display = 'none';
                    alert('Analysis Error: ' + (data.message || 'Detection failed. Please check audio integrity.'));
                }
            } catch (err) {
                if (loadingOverlay) loadingOverlay.style.display = 'none';
                console.error('Prediction request error:', err);
                alert('Connection error communicating with detection server: ' + err.message);
            }
        });
    }
}

// Quick Sample Loader for Demonstrations
async function loadDemoSample(filename) {
    const loadingOverlay = document.getElementById('loadingOverlay');
    if (loadingOverlay) loadingOverlay.style.display = 'flex';

    try {
        const response = await fetch(`/uploads/${filename}`);
        if (!response.ok) {
            throw new Error(`Demo file ${filename} not found on server.`);
        }
        const blob = await response.blob();
        const file = new File([blob], filename, { type: 'audio/wav' });

        const fileInput = document.getElementById('audioFileInput');
        if (fileInput) {
            const dataTransfer = new DataTransfer();
            dataTransfer.items.add(file);
            fileInput.files = dataTransfer.files;
            fileInput.dispatchEvent(new Event('change'));
        }
    } catch (err) {
        alert('Failed to load sample: ' + err.message);
    } finally {
        if (loadingOverlay) loadingOverlay.style.display = 'none';
    }
}

function formatBytes(bytes, decimals = 2) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const dm = decimals < 0 ? 0 : decimals;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
}

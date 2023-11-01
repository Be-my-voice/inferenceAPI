const prediction_label = document.getElementById('prediction')

document.getElementById('uploadButton').addEventListener('click', async () => {
    const fileInput = document.getElementById('videoInput');
    const videoPlayer = document.getElementById('videoPlayer');

    if (fileInput.files.length > 0) {
        const file = fileInput.files[0];
        const base64String = await convertVideoToBase64(file);

        videoPlayer.src = `data:video/mp4;base64,${base64String}`;
        videoPlayer.style.display = 'block';

    } else {
        alert('Please select a video file first.');
    }
});

document.getElementById('translate').addEventListener('click', async () => {
    const fileInput = document.getElementById('videoInput');
    const videoPlayer = document.getElementById('videoPlayer');

    if (fileInput.files.length > 0) {
        const file = fileInput.files[0];
        const base64String = await convertVideoToBase64(file);
        console.log(base64String)

        // Send the base64String to the server
        sendBase64ToServer(base64String);
    } else {
        alert('Please select a video file first.');
    }
});

function convertVideoToBase64(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = function (event) {
            resolve(event.target.result.split(',')[1]);
        };
        reader.onerror = function (error) {
            reject(error);
        };
        reader.readAsDataURL(file);
    });
}

function sendBase64ToServer(base64String) {
    fetch('http://127.0.0.1:8000/predict/video', {
        method: 'POST',
        body: JSON.stringify({ data: base64String }),
        headers: {
            'Content-Type': 'application/json',
        },
    })
        .then(response => response.json())
        .then(data => {
            console.log('Server response:', data);
            if(data.prediction != null) prediction_label.textContent = data.prediction;
        })
        .catch(error => {
            console.error('Error:', error);
        });
}

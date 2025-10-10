// Register Form Submission
document.getElementById('registerForm').addEventListener('submit', function(event) {
    event.preventDefault();
    const email = document.getElementById('email').value;
    const password = document.getElementById('password').value;
    const confirmPassword = document.getElementById('confirmPassword').value;

    if (password !== confirmPassword) {
        alert("Passwords do not match!");
        return;
    }

    fetch('/api/register', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ email, password })
    })
    .then(response => response.json())
    .then(data => {
        alert(data.message);
        window.location.href = "/login";
    })
    .catch(error => console.error('Error:', error));
});

// Login Form Submission
document.getElementById('loginForm').addEventListener('submit', function(event) {
    event.preventDefault();
    const email = document.getElementById('email').value;
    const password = document.getElementById('password').value;

    fetch('/api/login', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ email, password })
    })
    .then(response => response.json())
    .then(data => {
        if (data.code === 200) {
            window.location.href = "/upload_music";
        } else {
            alert(data.message);
        }
    })
    .catch(error => console.error('Error:', error));
});

// Upload Music Form Submission
document.getElementById('uploadForm').addEventListener('submit', function(event) {
    event.preventDefault();
    const file = document.getElementById('musicFile').files[0];

    const formData = new FormData();
    formData.append('file', file);

    fetch('/api/music/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('analysisResult').innerText = `Valence: ${data.valence}, Arousal: ${data.arousal}`;
    })
    .catch(error => console.error('Error:', error));
});

// Fetch Tasks
function fetchTasks() {
    fetch('/api/tasks')
    .then(response => response.json())
    .then(data => {
        const taskList = document.getElementById('taskList');
        data.tasks.forEach(task => {
            const li = document.createElement('li');
            li.classList.add('list-group-item');
            li.innerText = `Task ID: ${task.id}, Status: ${task.status}`;
            taskList.appendChild(li);
        });
    })
    .catch(error => console.error('Error:', error));
}

document.addEventListener('DOMContentLoaded', fetchTasks);

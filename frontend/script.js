document.getElementById("uploadForm").onsubmit = async (e) => {
    e.preventDefault();

    let file = document.getElementById("fileInput").files[0];

    if (!file) {
        alert("Please select a file!");
        return;
    }

    let formData = new FormData();
    formData.append("file", file);

    document.getElementById("loading").style.display = "block";

    let response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        body: formData
    });

    let data = await response.json();

    document.getElementById("loading").style.display = "none";

    document.getElementById("result").src = data.output;
};
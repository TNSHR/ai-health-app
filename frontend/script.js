async function predict() {
    let input = document.getElementById("inputData").value;

    let data = input.split(",").map(Number);

    let response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ data: data })
    });

    let result = await response.json();

    document.getElementById("result").innerText =
        "Prediction: " + result.prediction + " | Risk: " + result.risk;
}
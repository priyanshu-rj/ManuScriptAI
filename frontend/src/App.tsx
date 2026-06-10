import React, { useState } from "react";
import axios from "axios";
import "./App.css";

function App() {
  const [image, setImage] = useState<File | null>(null);
  const [preview, setPreview] = useState("");
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  const handleImage = (
    e: React.ChangeEvent<HTMLInputElement>
  ) => {
    if (!e.target.files) return;

    const file = e.target.files[0];

    setImage(file);
    setPreview(URL.createObjectURL(file));
    setResult(null);
  };

  const analyzeImage = async () => {
    if (!image) {
      alert("Please upload an image first.");
      return;
    }

    const formData = new FormData();
    formData.append("file", image);

    try {
      setLoading(true);

      const response = await axios.post(
        "http://127.0.0.1:8000/analyze",
        formData
      );

      setResult(response.data);
    } catch (error) {
      console.error(error);
      alert("Failed to connect to backend.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">

      <div className="header">
        <h1>Cultural Heritage Preservation</h1>
         <br />
        <p>
          AI-powered manuscript analysis using
          TrOCR, Language Detection, Script
          Classification and Century Prediction.
        </p>
      </div>

      <div className="upload-card">

    <label className="upload-label">
     Choose  Image

     <input
    type="file"
    accept="image/*"
    onChange={handleImage}
     />
    </label>

        {preview && (
          <img
            src={preview}
            alt="preview"
            className="preview"
          />
        )}

        <button
          onClick={analyzeImage}
          disabled={loading}
        >
          {loading
            ? "Analyzing..."
            : "Analyze Manuscript"}
        </button>

      </div>

      {result && (
        <div className="result-card">

          <h2>Analysis Result</h2>

          <div className="grid">

            <div className="info-box">
              <span>Language</span>
              <h3>{result.language}</h3>
            </div>

            <div className="info-box">
              <span>Script</span>
              <h3>{result.script}</h3>
            </div>

            <div className="info-box">
              <span>Century</span>
              <h3>
                {result.century || "Unknown"}
              </h3>
            </div>

          </div>

          <div className="ocr-box">
            <h3>OCR Extracted Text</h3>

            <textarea
              value={result.ocr_text}
              readOnly
              rows={8}
            />
          </div>

        </div>
      )}

    </div>
  );
}

export default App;
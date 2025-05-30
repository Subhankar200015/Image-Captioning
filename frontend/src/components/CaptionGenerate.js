import React, {useState} from 'react'
import { ReactComponent as UploadIcon } from '../assets/upload.svg'
import './CaptionGenerate.css'

function CaptionGenerate() {

  const[image, setImage] = useState(null)
  const[previewUrl, setPreviewUrl] = useState('')
  const[caption, setCaption] = useState('');

  const handleFileChange = (e) => {
    const file = e.target.files[0]
    if(file && file.type.startsWith("image/")) {
      setImage(file);
      setPreviewUrl(URL.createObjectURL(file))
    }
  }

  const generateCaption = async() => {

    if(!image) return;

    const formData = new FormData();
    formData.append('image', image)

    try {
      const response = await fetch('http://localhost:5000/caption',
        {
          method: 'POST',
          body: formData
        });

      const data = await response.json();
      setCaption(data.caption);
    }

    catch(error) {
      console.error('Error generating caption: ', error);
    }
  };

  return (
    <div className='container'>
        <h1>CaptionBot 3000: Your Image's Voice!</h1>
        <h2>Upload an image to generate a caption</h2>
        <input type="file" accept="image/*" id="fileInput" style={{display: 'none'}} onChange={handleFileChange}/>
        <label htmlFor="fileInput" className='custom-button'>Select Image</label>
        <br />
        {previewUrl && <img src={previewUrl} alt="preview" className='preview-image'/>}
        {image && (<button className='button' onClick={generateCaption}>Generate Caption</button>)}
        {caption && (
          <p><strong>Caption:</strong>{caption}</p>
        )}
    </div>
  )
}

export default CaptionGenerate
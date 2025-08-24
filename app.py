import gradio as gr
from implementation import predict_dog_breed

# Custom CSS for better styling
custom_css = """
.gradio-container {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
.title {
    text-align: center;
    color: #1f2937 ;
    font-size: 2.5em;
    margin-bottom: 20px;
    font-weight: bold;
}
.subtitle {
    text-align: center;
    color: #7f8c8d;
    font-size: 1.2em;
    margin-bottom: 30px;
}
.result-box {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    font-size: 1.3em;
    font-weight: bold;
    box-shadow: 0 8px 32px rgba(0,0,0,0.1);
}
.confidence-high {
    background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%);
}
.confidence-medium {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
}
.confidence-low {
    background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
}
"""

def predict_with_confidence(image):
    """Enhanced prediction function with input validation and better formatting"""
    try:
        
        breed,confidence = predict_dog_breed(image)
        
        # Parse the result to get breed and confidence
            
            # Add confidence threshold - if too low, suggest retry
        if confidence < 10:
            return f"""
            <div style="text-align: center; padding: 20px; color: #e74c3c;">
                <h3>⚠️ Low Confidence Prediction</h3>
                <p><strong>Breed:</strong> {breed.strip()}</p>
                <p><strong>Confidence:</strong> {confidence:.1f}%</p>
                <p><strong>This prediction has very low confidence.</strong></p>
                <p>Please try uploading a clearer image of a dog.</p>
            </div>
            """
            
            # Format the result for good predictions
        formatted_result = f"""
        <div style="text-align: center; padding: 20px;">
            <div style="font-size: 1.5em; margin-bottom: 15px;">🐕 <strong>Breed:</strong> {breed.strip()}</div>
            <div style="font-size: 1.3em; margin-bottom: 15px;">🎯 <strong>Confidence:</strong> {confidence:.1f}%</div>
            <div style="font-size: 1.2em; color: #666;">
                {'🌟 Excellent prediction!' if confidence >= 90 else '✅ Good prediction!' if confidence >= 70 else '🤔 Moderate confidence'}
            </div>
        </div>
        """
            
        return formatted_result
            
    except Exception as e:
        return f"❌ Error: {str(e)}"

with gr.Blocks(css=custom_css, title="Dog Breed Predictor") as iface:
    gr.HTML("""
        <div style="text-align: center; margin-bottom: 20px;">
            <h1 style="color: #ffffff; font-size: 2.5em; margin-bottom: 10px; font-weight: bold;">🐕 Dog Breed Predictor</h1>
            <p style="color: #7f8c8d; font-size: 1.2em; margin: 0;">Upload an image of a dog to identify its breed</p>
        </div>
    """)
    
    # Warning note about model behavior
    gr.HTML("""
        <div style="text-align: center; margin-bottom: 20px; padding: 15px; background: #ffebee; border: 2px solid #f44336; border-radius: 10px;">
            <p style="color: #d32f2f; font-size: 1.1em; font-weight: bold; margin: 0;">
                ⚠️ <strong>IMPORTANT:</strong> This model may give predictions for non-dog images as well. 
                For best results, please upload clear images of dogs only.
            </p>
        </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.HTML("""
                <h3 style="text-align: center; color: #3498db;">📸 Upload Image</h3>
            """)
            image_input = gr.Image(
                type="pil",
                label="",
                height=400,
                show_label=False
            )
            
            with gr.Row():
                clear_btn = gr.Button("🗑️ Clear", variant="secondary", size="lg")
                submit_btn = gr.Button("🚀 Predict Breed", variant="primary", size="lg")
        
        with gr.Column(scale=1):
            gr.HTML("""
                <h3 style="text-align: center; color: #e74c3c;">🎯 Prediction Result</h3>
            """)
            output = gr.HTML(
                value="<div style='text-align: center; color: #7f8c8d; font-size: 1.2em;'>Upload an image to get started!</div>",
                label="",
                show_label=False
            )
    
    gr.HTML("""
        <div style="text-align: center; margin-top: 30px; padding: 20px; background: #e8f4fd; border-radius: 10px; border: 2px solid #3498db;">
            <h4 style="color: #2c3e50;">💡 Tips for best results:</h4>
            <p style="color: #7f8c8d; margin: 10px 0;">
                • Use clear, well-lit images<br>
                • Ensure the dog is clearly visible<br>
                • Avoid blurry or dark photos<br>
                • Works best with front-facing shots
            </p>
        </div>
    """)
    
    submit_btn.click(
        fn=predict_with_confidence,
        inputs=image_input,
        outputs=output
    )
    
    clear_btn.click(
        fn=lambda: (None, "<div style='text-align: center; color: #7f8c8d; font-size: 1.2em;'>Upload an image to get started!</div>"),
        inputs=[],
        outputs=[image_input, output]
    )

if __name__ == "__main__":
    iface.launch(
        server_name="0.0.0.0",  # Allow external connections
        server_port=7860,       # Default Gradio port
        share=False             # Don't use Gradio's temporary sharing
    )
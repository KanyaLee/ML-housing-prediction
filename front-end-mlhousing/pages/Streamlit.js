export default function StreamlitPage() {
    return (
        <div className="text-center">
          <h1 className="text-2xl font-bold mb-4">Our Machine Learning Model</h1>
          <p className="mb-4">Click the button below to access our Streamlit app.</p>
          
          <a 
            href="https://ml-housing-prediction-bangkok.streamlit.app/" 
            className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded" 
            target='_blank' 
            rel="noopener noreferrer"
          >
            Go to Streamlit App
          </a>
    
        </div>
    );
}

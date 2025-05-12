from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from predict_price import predict_price

# Create FastAPI Instance
app = FastAPI(
    title="HDB Price Prediction API",
    description=(
        "Predicting resale prices of HDB flats using a pre-trained LightGBM model. "
        "Provide flat attributes and receive a price estimate in SGD."
    ),
    version="1.2.0",
    contact={"name": "Dillon", "email": "dillonchong01@gmail.com"},
)

# Define Request Schema using Pydantic BaseModel
class PredictRequest(BaseModel):
    Flat_Type: str = Field(..., description="Type of Flat (E.g. 3 Room, Executive, Multi-Gen)")
    Storey: int = Field(..., ge=0, description="Storey Number (0+)")
    Floor_Area: float = Field(..., gt=0, description="Floor Area in sqm (>0)")
    Remaining_Lease: float = Field(..., ge=0, description="Remaining Lease in Years (>=0)")
    RPI: float = Field(..., ge=0, description="Resale Price Index in Current Quarter")
    Address: str = Field(..., description="Address of Flat (E.g. 718 Yishun St 33)")
    Town: str = Field(..., description="Town Name (E.g. Tampines, Ang Mo Kio)")

# Define Response Schema
class PredictResponse(BaseModel):
    price: float = Field(..., description="Predicted Resale Price in SGD")

# Home Endpoint
@app.get(
    "/",
    response_class=HTMLResponse,
    summary="Homepage",
    include_in_schema=False
)
async def homepage():
    """
    Simple HTML homepage with a welcome message and basic usage instructions.
    """
    html_content = """
    <html>
        <head><title>HDB Price Prediction API</title></head>
        <body>
            <h1>Welcome to the HDB Resale Price Prediction API Service</h1>
            <p>Use the <a href='/docs'>Swagger UI</a> to explore the API endpoints.</p>
            <p>POST your data to the <code>/predict</code> endpoint to receive HDB Resale Price Predictions via the REST API.</p>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=200)

# Health-Check Endpoint to keep Container Warm
@app.get("/health")
async def health_check():
    """
    Simple endpoint to check service health. Front-end can ping this to warm up the model
    """
    return {"status": "ok"}

# Prediction Endpoint
@app.post("/predict",
          response_model=PredictResponse,
          summary="Predict HDB Resale Price",
          response_description="The predicted resale price in SGD")

async def predict_endpoint(request: PredictRequest):
    """
     Receive flat attributes and return a price prediction.
    """
    try:
        # Convert Pydantic model to dict for processing
        input_data = request.dict()
        # Get prediction from your model
        predicted_price = predict_price(input_data)
        return PredictResponse(price=predicted_price)

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
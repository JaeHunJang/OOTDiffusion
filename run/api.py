import shutil
from typing import Optional
from fastapi import FastAPI, HTTPException, Form
from fastapi.responses import JSONResponse
import subprocess
import os
from pathlib import Path
import uuid
import requests
from io import BytesIO
from PIL import Image
import base64

app = FastAPI()

@app.get("/")
def read_root():
    return {"health": "Good"}

def open_image_from_url(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return img

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

@app.post("/generate")
async def process(
        memberId: int = Form(...),
        modelImagePath: str = Form(...),
        clothImagePath: str = Form(...),
        modelType: Optional[str] = Form("dc"),
        category: int = Form(...),
        scale: Optional[float] = Form(2.0),
        sample: Optional[int] = Form(4)
):
    try:
        output_dir = Path(f"./images_output/{memberId}")
        output_dir.mkdir(parents=True, exist_ok=True)

        unique_id = str(uuid.uuid4())
        temp_dir = output_dir / unique_id
        temp_dir.mkdir(parents=True, exist_ok=True)

        # 절대 경로로 설정
        script_path = Path(__file__).parent / "run_ootd.py"

        # GPU 환경 변수 설정
        os.environ['CUDA_VISIBLE_DEVICES'] = '9'

        # run_ootd.py 스크립트를 subprocess를 통해 실행
        command = [
            "python", str(script_path),
            "--model_path", modelImagePath,
            "--cloth_path", clothImagePath,
            "--model_type", modelType if modelType else "dc",
            "--category", str(category),
            "--scale", str(scale),
            "--sample", str(sample),
            "--outputDir", str(temp_dir)
        ]
        print(f"Running command: {' '.join(command)} with GPU 9")
        result = subprocess.run(command, capture_output=True, text=True, cwd=str(script_path.parent))

        # 명령어 실행 결과 출력
        print(f"args: {result.args}")
        print(f"returncode: {result.returncode}")
        print(f"stdout: {result.stdout}")
        print(f"stderr: {result.stderr}")

        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=result.stderr)

        # 이미지 경로 목록 추출
        image_paths = []
        for line in result.stdout.split('\n'):
            if "Generated images:" in line:
                image_paths = line.split("Generated images:")[1].strip().strip("[]").replace("'", "").split(", ")

        if not image_paths:
            raise HTTPException(status_code=500, detail="No images generated")

        # 각 이미지를 base64로 인코딩
        encoded_images = [encode_image_to_base64(Path(image_path.strip())) for image_path in image_paths]

        # base64 인코딩된 이미지를 JSON으로 반환
        response = JSONResponse(content={"images": encoded_images})

        # 임시 디렉토리 삭제
        shutil.rmtree(temp_dir)

        return response

    except Exception as e:
        # 예외 발생 시에도 임시 디렉토리 삭제
        shutil.rmtree(temp_dir)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

import subprocess
from fastapi import FastAPI, HTTPException, Form
from fastapi.responses import JSONResponse
from pathlib import Path
import uuid
import shutil
import base64
import requests
from PIL import Image
from io import BytesIO
import traceback
# import os
# os.environ['CUDA_VISIBLE_DEIVCES']='9'

app = FastAPI()

@app.get("/")
def read_root():
    return {"health": "Good"}

def encode_image_to_base64(image_path):
    if not image_path.exists():
        raise FileNotFoundError(f"The file {image_path} does not exist.")
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def download_and_save_image(url, output_dir, filename):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))

    # RGBA 모드인 경우 RGB로 변환
    if img.mode == 'RGBA':
        img = img.convert('RGB')

    image_path = output_dir / filename
    img.save(image_path)
    return image_path

@app.post("/generate")
async def process(
        memberId: int = Form(...),
        modelImagePath: str = Form(...),
        clothImagePath: str = Form(...),
        modelType: str = Form("dc"),
        category: int = Form(...),
        scale: float = Form(2.0),
        sample: int = Form(1)
):
    try:
        output_dir = Path(f"./images_output/{memberId}")
        output_dir.mkdir(parents=True, exist_ok=True)

        unique_id = str(uuid.uuid4())
        temp_dir = output_dir / unique_id
        temp_dir.mkdir(parents=True, exist_ok=True)

        # 이미지를 다운로드하고 로컬에 저장
        model_img_path = download_and_save_image(modelImagePath, temp_dir, "model_img.jpg")
        cloth_img_path = download_and_save_image(clothImagePath, temp_dir, "cloth_img.jpg")

        print(f"Running run_ootd with modelType={modelType}, category={category}, scale={scale}, sample={sample}")

        # subprocess로 run_ootd.py 스크립트 실행
        command = [
            "python", "run_ootd_custom.py",  # `run_ootd.py`의 실제 경로로 바꾸세요
            "--gpu_id", "9",
            "--model_path", str(model_img_path),
            "--cloth_path", str(cloth_img_path),
            "--model_type", modelType,
            "--category", str(category),
            "--scale", str(scale),
            "--step", "20",
            "--sample", str(sample),
            "--seed", "-1",
            "--output_dir", str(temp_dir)
        ]

        result = subprocess.run(command, capture_output=True, text=True)

        if result.returncode != 0:
            raise Exception(f"Subprocess failed: {result.stderr}")

        # 출력에서 이미지 경로를 파싱
        output_lines = result.stdout.strip().splitlines()
        image_paths = []
        for line in output_lines:
            if line.startswith("Generated images:"):
                paths_str = line.split("Generated images:")[1].strip()
                paths = paths_str.strip("[]").replace("'", "").split(", ")
                image_paths.extend(paths)

        if not image_paths:
            raise HTTPException(status_code=500, detail="No images generated")

        # 각 이미지를 base64로 인코딩
        encoded_images = []
        for image_path in image_paths:
            img_path = Path(image_path.strip())
            if img_path.exists():
                encoded_images.append(encode_image_to_base64(img_path))
            else:
                print(f"Warning: {img_path} does not exist.")

        # base64 인코딩된 이미지를 JSON으로 반환
        response = JSONResponse(content={"images": encoded_images})

        # 임시 디렉토리 삭제
        shutil.rmtree(temp_dir)

        return response

    except Exception as e:
        print("Error occurred:", str(e))
        traceback.print_exc()

        # 예외 발생 시에도 임시 디렉토리 삭제
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

from fastapi import FastAPI, Response, Body, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Union, Literal
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import io
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# --- CORS 설정 (이전과 동일) ---
origins = [
    "http://13.236.16.220:4002",
    "http://localhost:4002"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# --- CORS 설정 끝 ---

# --- Pydantic 모델 정의 (이전과 동일) ---
class RectangleFurniture(BaseModel):
    type: Literal["rectangle"] = "rectangle"
    x1: float = Field(..., description="직사각형의 첫 번째 대각선 꼭짓점 X좌표 (mm).")
    y1: float = Field(..., description="직사각형의 첫 번째 대각선 꼭짓점 Y좌표 (mm).")
    x2: float = Field(..., description="직사각형의 두 번째 대각선 꼭짓점 X좌표 (mm).")
    y2: float = Field(..., description="직사각형의 두 번째 대각선 꼭짓점 Y좌표 (mm).")
    color: str = Field("blue", description="가구 색상 (CSS 색상 이름 또는 HEX 코드).")
    label: str = Field("직사각형 가구", description="가구 라벨.")

class CircleFurniture(BaseModel):
    type: Literal["circle"] = "circle"
    center_x: float = Field(..., description="원의 중심 X좌표 (mm).")
    center_y: float = Field(..., description="원의 중심 Y좌표 (mm).")
    radius: float = Field(..., gt=0, description="원의 반지름 (mm, 0보다 커야 함).")
    color: str = Field("red", description="가구 색상 (CSS 색상 이름 또는 HEX 코드).")
    label: str = Field("원형 가구", description="가구 라벨.")

Furniture = Union[RectangleFurniture, CircleFurniture]

# --- 새로운 Pydantic 모델 정의: 요청 본문의 구조를 명확히 합니다 ---
# React가 {"furniture": [...]} 형태로 데이터를 보내므로, 이 구조에 맞춥니다.
class RoomPlanRequest(BaseModel):
    furniture: List[Furniture] = Field(..., description="방에 배치할 가구 목록.")
    # 만약 나중에 width나 height도 본문에 포함시키고 싶다면 여기에 추가할 수 있습니다.
    # width: float = Field(..., description="방의 가로 길이 (mm).")
    # height: float = Field(..., description="방의 세로 길이 (mm).")


# --- FastAPI 엔드포인트 수정 ---
@app.post("/room-plan/")
async def create_room_plan_with_furniture(
    # 방의 너비와 높이는 여전히 URL 쿼리 파라미터로 받습니다.
    width: float = Query(..., gt=0, description="방의 가로 길이 (mm, 0보다 커야 함)."),
    height: float = Query(..., gt=0, description="방의 세로 길이 (mm, 0보다 커야 함)."),
    # 요청 본문은 이제 RoomPlanRequest 모델에 따라 파싱됩니다.
    request_body: RoomPlanRequest = Body(..., description="평면도 생성을 위한 요청 본문 데이터.")
):
    """
    방 크기와 가구 목록을 받아 2D 평면도를 PNG 이미지로 생성합니다.
    방과 가구의 좌표는 (0,0)을 기준으로 합니다.
    """
    # 요청 본문에서 가구 목록을 추출합니다.
    furniture_to_draw = request_body.furniture

    if width <= 0 or height <= 0:
        raise HTTPException(status_code=400, detail="가로와 세로 길이는 양수여야 합니다.")

    fig = Figure(figsize=(width / 100, height / 100))
    ax = fig.add_subplot(111)

    room_rectangle = plt.Rectangle((0, 0), width, height,
                                   edgecolor='black', facecolor='lightgray',
                                   linewidth=2, label=f"방 ({width}x{height} mm)")
    ax.add_patch(room_rectangle)

    # --- 가구 그리기: 이제 furniture_to_draw를 사용합니다 ---
    for item in furniture_to_draw:
        if item.type == "rectangle":
            x_min = min(item.x1, item.x2)
            y_min = min(item.y1, item.y2)
            rect_width = abs(item.x2 - item.x1)
            rect_height = abs(item.y2 - item.y1)

            if not (0 <= x_min and x_min + rect_width <= width and
                    0 <= y_min and y_min + rect_height <= height):
                print(f"경고: 직사각형 가구 '{item.label}'가 방 경계를 벗어났습니다. 그리지 않습니다.")
                continue

            rect_furniture = plt.Rectangle((x_min, y_min), rect_width, rect_height,
                                           edgecolor='black', facecolor=item.color,
                                           alpha=0.7, label=item.label)
            ax.add_patch(rect_furniture)
            ax.text(x_min + rect_width / 2, y_min + rect_height / 2, item.label,
                    ha='center', va='center', fontsize=8, color='white', weight='bold')

        elif item.type == "circle":
            if not (0 <= item.center_x - item.radius and item.center_x + item.radius <= width and
                    0 <= item.center_y - item.radius and item.center_y + item.radius <= height):
                print(f"경고: 원형 가구 '{item.label}'가 방 경계를 벗어났습니다. 그리지 않습니다.")
                continue

            circle_furniture = plt.Circle((item.center_x, item.center_y), item.radius,
                                          edgecolor='black', facecolor=item.color,
                                          alpha=0.7, label=item.label)
            ax.add_patch(circle_furniture)
            ax.text(item.center_x, item.center_y, item.label,
                    ha='center', va='center', fontsize=8, color='white', weight='bold')

    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_xlabel(f"가로 (mm)")
    ax.set_ylabel(f"세로 (mm)")
    ax.set_title(f"방 평면도: {width}mm x {height}mm (가구 포함)")
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True)
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1))

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight', dpi=300)
    buf.seek(0)
    plt.close(fig)

    return Response(content=buf.getvalue(), media_type="image/png")

# --- Uvicorn 서버 실행 설정 (이전과 동일) ---
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=3002, reload=True)
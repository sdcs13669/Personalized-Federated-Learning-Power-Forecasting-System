"""演示数据集清单（数据源 = GitHub raw URL）。"""
from fastapi import APIRouter, Depends

from server.routers.auth import get_current_user_from_header
from server.models import User

router = APIRouter(prefix="/api/datasets", tags=["datasets"])

# 数据源清单：Task 1 上传后把真实 URL 填到这里
DATASETS = [
    {"id": "steel_ind", "name": "钢厂用电（整份）",
     "client_id": "steel_ind_0",
     "url": "https://raw.githubusercontent.com/<owner>/<repo>/main/data/app_datasets/steel_ind.csv.zip",
     "description": "steel 数据集整份，30 分钟粒度用电负荷",
     "size": "待填"},
    {"id": "tetouan_0", "name": "城市用电 - 区域 1",
     "client_id": "tetouan_0",
     "url": "https://raw.githubusercontent.com/<owner>/<repo>/main/data/app_datasets/tetouan_zone0.csv.zip",
     "description": "Tetouan 城市用电区域 1 序列",
     "size": "待填"},
    {"id": "tetouan_1", "name": "城市用电 - 区域 2",
     "client_id": "tetouan_1",
     "url": "https://raw.githubusercontent.com/<owner>/<repo>/main/data/app_datasets/tetouan_zone1.csv.zip",
     "description": "Tetouan 城市用电区域 2 序列",
     "size": "待填"},
    {"id": "tetouan_2", "name": "城市用电 - 区域 3",
     "client_id": "tetouan_2",
     "url": "https://raw.githubusercontent.com/<owner>/<repo>/main/data/app_datasets/tetouan_zone2.csv.zip",
     "description": "Tetouan 城市用电区域 3 序列",
     "size": "待填"},
]


@router.get("")
def list_datasets(user: User = Depends(get_current_user_from_header)):
    return DATASETS

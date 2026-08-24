"""演示数据集清单（数据源 = GitHub raw URL）。"""
from fastapi import APIRouter, Depends

from server.routers.auth import get_current_user_from_header
from server.models import User

router = APIRouter(prefix="/api/datasets", tags=["datasets"])

# 数据源清单：与 app/agent.py、data/app_datasets/README.md（Task 1）对齐
DATASETS = [
    {"id": "steel_ind_0", "name": "钢铁厂用电（整份）",
     "client_id": "steel_ind_0",
     "url": "https://raw.githubusercontent.com/sdcs13669/Personalized-Federated-Learning-Power-Forecasting-System/main/data/app_datasets/steel_ind_0.zip",
     "description": "钢铁厂用电，整份（365 天，30 分钟粒度）",
     "size": "386K"},
    {"id": "tetouan_0", "name": "城市用电 - Zone1（工业区）",
     "client_id": "tetouan_city_0",
     "url": "https://raw.githubusercontent.com/sdcs13669/Personalized-Federated-Learning-Power-Forecasting-System/main/data/app_datasets/tetouan_0.zip",
     "description": "Tetouan 城市用电 Zone1（工业区，364 天）",
     "size": "540K"},
    {"id": "tetouan_1", "name": "城市用电 - Zone2（混合区）",
     "client_id": "tetouan_city_1",
     "url": "https://raw.githubusercontent.com/sdcs13669/Personalized-Federated-Learning-Power-Forecasting-System/main/data/app_datasets/tetouan_1.zip",
     "description": "Tetouan 城市用电 Zone2（混合区，364 天）",
     "size": "542K"},
    {"id": "tetouan_2", "name": "城市用电 - Zone3（居民区）",
     "client_id": "tetouan_city_2",
     "url": "https://raw.githubusercontent.com/sdcs13669/Personalized-Federated-Learning-Power-Forecasting-System/main/data/app_datasets/tetouan_2.zip",
     "description": "Tetouan 城市用电 Zone3（居民区，364 天）",
     "size": "541K"},
]


@router.get("")
def list_datasets(user: User = Depends(get_current_user_from_header)):
    return DATASETS

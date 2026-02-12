from fastapi import FastAPI, HTTPException
import uvicorn
import logging
import time
from typing import List

# 导入predict模块中的函数
from predict import text_classify_bert, text_classify_bert_batch

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="BERT文本分类API",
    description="基于BERT的文本分类服务，只返回最高概率的类别",
    version="1.0.0",
    docs_url="/docs"
)


@app.middleware("http")
async def log_requests(request, call_next):
    """日志中间件"""
    start_time = time.time()
    logger.info(f"{request.method} {request.url.path}")

    response = await call_next(request)

    process_time = time.time() - start_time
    logger.info(f"完成 - 状态: {response.status_code} - 耗时: {process_time:.2f}s")

    response.headers["X-Process-Time"] = str(process_time)
    return response


@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "BERT文本分类API",
        "description": "只返回最高概率的类别",
        "endpoints": {
            "分类单个文本": "GET /classify/{text}",
            "批量分类": "POST /classify/batch"
        }
    }


@app.get("/classify/{text}")
async def classify_text(text: str):
    """
    分类单个文本
    Parameters:
    -----------
    text : str
        要分类的文本
    Returns:
    --------
    dict : {
        "text": str,            # 原始文本
        "predicted_class": int, # 预测的类别
        "category": str,        # 类别名称
        "confidence": float     # 置信度
    }
    """
    logger.info(f"分类请求: {text[:50]}...")

    try:
        result = text_classify_bert(text)
        result["text"] = text  # 添加原始文本

        # 检查是否预测失败
        if result["predicted_class"] == -1:
            raise HTTPException(status_code=500, detail=result.get("error", "预测失败"))

        logger.info(f"分类结果: 类别={result['predicted_class']}, 置信度={result['confidence']:.2%}")
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"分类失败: {e}")
        raise HTTPException(status_code=500, detail=f"分类失败: {str(e)}")


from pydantic import BaseModel


class BatchRequest(BaseModel):
    texts: List[str]


@app.post("/classify/batch")
async def classify_batch(request: BatchRequest):
    """
    批量分类文本

    Parameters:
    -----------
    request.texts : List[str]
        要分类的文本列表

    Returns:
    --------
    dict : {
        "count": int,       # 总文本数
        "results": list     # 每个文本的预测结果
    }
    """
    if len(request.texts) > 100:
        raise HTTPException(status_code=400, detail="一次最多处理100条文本")

    logger.info(f"批量分类请求: {len(request.texts)}条文本")

    try:
        results = text_classify_bert_batch(request.texts)

        # 统计成功/失败
        success_count = sum(1 for r in results if r["predicted_class"] != -1)

        logger.info(f"批量分类完成: 成功{success_count}/{len(results)}")

        return {
            "count": len(results),
            "success_count": success_count,
            "results": results
        }

    except Exception as e:
        logger.error(f"批量分类失败: {e}")
        raise HTTPException(status_code=500, detail=f"批量分类失败: {str(e)}")


@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "running",
        "service": "BERT文本分类API"
    }


if __name__ == "__main__":
    logger.info("启动BERT文本分类API服务...")
    logger.info("访问 http://localhost:8000 进行测试")
    logger.info("访问 http://localhost:8000/docs 查看API文档")

    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # 代码修改自动重启
        log_level="debug",  # uvicorn 详细日志
        reload_dirs=["."],  # 监控当前目录
        reload_excludes=["*.tmp"]  # 排除临时文件
    )
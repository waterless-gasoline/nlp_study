from fastapi import FastAPI, Request, HTTPException
import uvicorn
import logging
import time

from starlette.responses import JSONResponse
from 戴长瑞.week01.classification_knn import text_classify_using_ml_mao, text_classify_using_ml,text_classify_cnn

# 配置日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI()


# 调试中间件
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    request_id = str(start_time).replace('.', '')

    logger.info(f"[{request_id}] {request.method} {request.url}")
    logger.debug(f"[{request_id}] Headers: {dict(request.headers)}")

    # 如果是 POST/PUT，记录请求体（小心处理大文件）
    if request.method in ["POST", "PUT"]:
        try:
            body = await request.body()
            if len(body) < 1000:  # 只记录小请求体
                logger.debug(f"[{request_id}] Body: {body}")
        except:
            pass

    response = await call_next(request)

    process_time = time.time() - start_time
    logger.info(f"[{request_id}] 完成 - 状态: {response.status_code} - 耗时: {process_time:.2f}s")

    # 添加调试头部
    response.headers["X-Process-Time"] = str(process_time)
    response.headers["X-Request-ID"] = request_id

    return response


@app.get("/")
async def root():
    """根路径 - 这里可以加断点"""
    logger.debug("进入 root 函数")

    # 模拟一些处理逻辑
    data = {"Hello": "World"}

    # 调试：检查数据
    logger.debug(f"准备返回数据: {data}")

    return data


@app.get("/{type}/{text}")
async def read_item(type: str, text: str, q: str = None):
    """
    根据type调用不同的文本分类函数

    Parameters:
    -----------
    type : str
        分类方法类型，支持: 'KNN', 'CNN'
    text : str
        要分类的文本
    q : str, optional
        额外的查询参数

    Returns:
    --------
    dict : 分类结果
    """

    # 记录日志
    logger.info(f"接收请求 - 类型: {type}, 文本: {text}, 查询参数: {q}")

    # 根据type选择不同的分类函数
    if type.upper() == "KNN":
        result = text_classify_using_ml_mao(text)
    elif type.upper() == "CNN":
        result = text_classify_cnn(text)
    else:
        logger.warning(f"不支持的类型: {type}")
        raise HTTPException(
            status_code=400,
            detail=f"不支持的类型: {type}。请使用 'KNN' 或 'CNN'"
        )

    # 添加额外的查询参数到结果中（如果存在）
    if q:
        result["query"] = q

    logger.info(f"处理完成 - 类型: {type}, 结果: {result}")
    return result


@app.get("/error")
async def trigger_error():
    """触发错误用于调试"""
    logger.warning("触发错误端点")
    return {"message": "这行不会执行"}


@app.exception_handler(Exception)
async def debug_exception_handler(request: Request, exc: Exception):
    """自定义错误处理，便于调试"""
    logger.error(f"捕获异常: {type(exc).__name__}: {exc}")
    logger.debug(f"异常详情: {exc.__traceback__}")

    return JSONResponse(
        status_code=500,
        content={
            "error": str(exc),
            "type": type(exc).__name__,
            "path": str(request.url)
        }
    )


if __name__ == "__main__":
    logger.info("启动调试模式 FastAPI 服务器...")
    logger.info("访问 http://localhost:8000 进行测试")
    logger.info("访问 http://localhost:8000/docs 查看文档")

    uvicorn.run(
        "fastapi_bulid:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # 代码修改自动重启
        log_level="debug",  # uvicorn 详细日志
        reload_dirs=["."],  # 监控当前目录
        reload_excludes=["*.tmp"]  # 排除临时文件
    )
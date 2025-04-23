# 图像检索

本示例改编自官方教程 [Image Search with Milvus](https://milvus.io/docs/image_similarity_search.md)。

在 JupyterLab / CodeServer 的终端，安装依赖并准备数据：

```bash
pip install --upgrade pymilvus timm
wget https://github.com/milvus-io/pymilvus-assets/releases/download/imagedata/reverse_image_search.zip
unzip -q -o reverse_image_search.zip
```

修改 Python 脚本中的 Milvus URI 等全局变量，然后执行：

```bash
python image-search.py
```

## 预期输出

```bash
Query Image: ./test/Afghan_hound/n02088094_4261.JPEG

Top 10 Results:
1. ./train/Afghan_hound/n02088094_5911.JPEG
2. ./train/Afghan_hound/n02088094_6533.JPEG
3. ./train/Afghan_hound/n02088094_2164.JPEG
4. ./train/Bouvier_des_Flandres/n02106382_5429.JPEG
5. ./train/Bouvier_des_Flandres/n02106382_6653.JPEG
6. ./train/Afghan_hound/n02088094_6565.JPEG
7. ./train/Afghan_hound/n02088094_1045.JPEG
8. ./train/Afghan_hound/n02088094_5532.JPEG
9. ./train/Afghan_hound/n02088094_7360.JPEG
10. ./train/soft-coated_wheaten_terrier/n02098105_400.JPEG
```

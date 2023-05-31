## 用户手册

### 软件信息

* 版本：2.0
* 作者：lh9171338
* 日期：2022-02-20

### 快捷键（默认）

* Ctrl + O：打开图像文件夹
* Ctrl + S：保存标注结果
* Ctrl + V：进入下一张图像
* Ctrl + B：返回上一张图像
* Ctrl + C：进入线段标注状态
* Ctrl + D：删除选中的线段标注
* Ctrl + U：查看用户手册

**注**：编辑`default.yaml`文件可以修改快捷键

### 鼠标按键

* 左键：标注一个线段端点
* 右键：进入或退出线段标注状态

### 操作流程

1. 选择一个待标注图像所在的文件夹
2. 单击鼠标右键（或通过`Ctrl + C`快捷键）进入线段标注状态
3. 通过鼠标左键标注线段端点，标注完两个线段端点后，一条线段标注完成，中途可以单击鼠标右键（或通过`Ctrl + C`快捷键）退出线段标注状态
4. 重复步骤2-3，中途最好多次保存标注结果，以防程序意外崩溃

### 数据集文件结构

    |-- dataset   
        |-- <image folder>
            |-- 000001.png  
            |-- 000002.png  
            |-- ...  
        |-- <label folder>  
            |-- 000001.mat  
            |-- 000002.mat  
            |-- ...  
        |-- <coeff folder>
            |-- 000001.yaml
            |-- 000002.yaml  
            |-- ...

**注**：`<image folder>`、`<label folder>`和`<coeff folder>`路径在`default.yaml`文件中配置，三者可以相同，`<coeff folder>`路径并非必须的

### 程序运行

```shell
python Labelline.py --type <image type> [--coeff_file <coeff image>]  # type = 0: 平面图像, 1: 鱼眼图像, 2: 球面图像
```
**注**：平面图像和球面图像不需要提供相机参数，鱼眼图像需要提供相机参数，如果不使用`--coeff_file`命令显式指定相机参数，程序则会在`coeff_folder`路径下获取图像对应的相机参数

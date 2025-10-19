import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import time
import io
import base64
import torchvision.transforms as transforms

# 页面配置
st.set_page_config(
    page_title="动物保护 AI 识别系统",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS美化界面 - 更适合初中生
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #2E8B57;
    text-align: center;
    margin-bottom: 2rem;
    font-weight: bold;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
}
.sub-header {
    font-size: 1.8rem;
    color: #228B22;
    margin-top: 1.5rem;
    margin-bottom: 1rem;
    border-bottom: 3px solid #90EE90;
    padding-bottom: 0.5rem;
    background: linear-gradient(90deg, #E8F5E8, transparent);
    padding: 10px;
    border-radius: 10px;
}
.mission-card {
    background: linear-gradient(135deg, #F0FFF0, #E0F7E0);
    padding: 1.5rem;
    border-radius: 15px;
    border-left: 8px solid #32CD32;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}
.story-box {
    background: linear-gradient(135deg, #E6F3FF, #D4EBFF);
    padding: 1.5rem;
    border-radius: 15px;
    border-left: 8px solid #1E90FF;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}
.success-box {
    background: linear-gradient(135deg, #E8F5E8, #D4F0D4);
    padding: 1rem;
    border-radius: 12px;
    border: 2px solid #32CD32;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.warning-box {
    background: linear-gradient(135deg, #FFF3CD, #FFE8A3);
    padding: 1rem;
    border-radius: 12px;
    border: 2px solid #FFC107;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.animal-card {
    background: linear-gradient(135deg, #FFF8DC, #FFEFB3);
    padding: 1.5rem;
    border-radius: 15px;
    margin: 0.5rem;
    text-align: center;
    box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    transition: transform 0.3s ease;
    border: 2px solid #FFD700;
}
.animal-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 8px 16px rgba(0,0,0,0.2);
}
.training-log {
    background-color: #F5F5F5;
    padding: 1rem;
    border-radius: 12px;
    font-family: monospace;
    max-height: 300px;
    overflow-y: auto;
    border: 2px solid #DDD;
}
.column-section {
    background: linear-gradient(135deg, #F8F9FA, #E9ECEF);
    padding: 1.5rem;
    border-radius: 15px;
    border: 2px solid #DEE2E6;
    margin-bottom: 1.5rem;
    height: 100%;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}
.upload-help {
    font-size: 0.8rem;
    color: #666;
    margin-top: 0.5rem;
}
.learning-sheet {
    background: linear-gradient(135deg, #FFF3E0, #FFE0B2);
    padding: 2rem;
    border-radius: 20px;
    border: 3px solid #FF9800;
    margin: 1rem 0;
    box-shadow: 0 6px 12px rgba(0,0,0,0.15);
}
.learning-question {
    background: white;
    padding: 1.5rem;
    border-radius: 12px;
    margin: 1rem 0;
    border-left: 5px solid #4CAF50;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.step-box {
    background: linear-gradient(135deg, #E3F2FD, #BBDEFB);
    padding: 1.5rem;
    border-radius: 12px;
    margin: 1rem 0;
    border: 2px solid #2196F3;
}
.fun-fact {
    background: linear-gradient(135deg, #FCE4EC, #F8BBD0);
    padding: 1rem;
    border-radius: 12px;
    margin: 0.5rem 0;
    border-left: 5px solid #E91E63;
    font-style: italic;
}
.animal-feature {
    background: linear-gradient(135deg, #E8F5E9, #C8E6C9);
    padding: 1rem;
    border-radius: 10px;
    margin: 0.5rem 0;
    border-left: 4px solid #4CAF50;
}
.progress-info {
    background: linear-gradient(135deg, #E1F5FE, #B3E5FC);
    padding: 1rem;
    border-radius: 10px;
    margin: 1rem 0;
    border-left: 4px solid #03A9F4;
}
</style>
""", unsafe_allow_html=True)

# 应用标题
st.markdown('<div class="main-header">🐾 动物保护 AI 识别系统 🐾</div>', unsafe_allow_html=True)

# 侧边栏导航
st.sidebar.title("导航")
page = st.sidebar.radio("选择任务阶段:",
                        ["项目介绍",
                         "学习单",
                         "第一阶段：物种分类系统",
                         "第二阶段：个体识别系统",
                         "AI研究员证书"])

# 初始化session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'training_history' not in st.session_state:
    st.session_state.training_history = None
if 'class_names_phase1' not in st.session_state:
    st.session_state.class_names_phase1 = []
if 'class_names_phase2' not in st.session_state:
    st.session_state.class_names_phase2 = []
if 'device' not in st.session_state:
    st.session_state.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if 'train_loader' not in st.session_state:
    st.session_state.train_loader = None
if 'trained_phase1' not in st.session_state:
    st.session_state.trained_phase1 = False
if 'trained_phase2' not in st.session_state:
    st.session_state.trained_phase2 = False
if 'class_images' not in st.session_state:
    st.session_state.class_images = {}
if 'class_images_phase2' not in st.session_state:
    st.session_state.class_images_phase2 = {}
if 'uploaded_files_cache' not in st.session_state:
    st.session_state.uploaded_files_cache = {}
if 'learning_answers' not in st.session_state:
    st.session_state.learning_answers = {}


# 自定义数据集类
class AnimalDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


# 定义更快的CNN模型 - 显著减少参数数量
class FastAnimalCNN(nn.Module):
    def __init__(self, num_classes):
        super(FastAnimalCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # 动态计算全连接层输入尺寸
        self.fc_input_size = self._get_fc_input_size()

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.fc_input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def _get_fc_input_size(self):
        # 通过一个虚拟输入来计算全连接层的输入尺寸
        x = torch.zeros(1, 3, 64, 64)
        x = self.conv_layers(x)
        return x.view(1, -1).size(1)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


# 改进的训练函数 - 优化训练速度
def train_model(model, train_loader, criterion, optimizer, epochs, device):
    train_losses = []
    train_accs = []

    # 创建进度条和状态文本
    progress_bar = st.progress(0)
    status_text = st.empty()
    time_placeholder = st.empty()

    start_time = time.time()

    for epoch in range(epochs):
        # 更新进度条
        progress = (epoch + 1) / epochs
        progress_bar.progress(progress)

        # 训练阶段
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # 计算预计剩余时间
        elapsed_time = time.time() - start_time
        time_per_epoch = elapsed_time / (epoch + 1)
        remaining_time = time_per_epoch * (epochs - epoch - 1)

        status_text.text(f"训练中... 第 {epoch + 1}/{epochs} 轮")
        time_placeholder.markdown(f"""
        <div class="progress-info">
        <strong>进度:</strong> {epoch + 1}/{epochs} 轮<br>
        <strong>当前准确率:</strong> {train_acc:.2f}%<br>
        <strong>预计剩余时间:</strong> {remaining_time:.1f}秒
        </div>
        """, unsafe_allow_html=True)

    # 清除状态文本
    status_text.text("训练完成！")
    progress_bar.empty()
    time_placeholder.empty()

    total_time = time.time() - start_time
    st.success(f"训练完成！总耗时: {total_time:.1f}秒")

    return {
        'train_loss': train_losses,
        'train_acc': train_accs
    }


# 绘制训练历史
def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # 准确率
    ax1.plot(history['train_acc'], label='训练准确率', color='#4CAF50', linewidth=2)
    ax1.set_title('模型准确率', fontsize=14, fontweight='bold')
    ax1.set_xlabel('训练轮次')
    ax1.set_ylabel('准确率 (%)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 损失
    ax2.plot(history['train_loss'], label='训练损失', color='#FF5722', linewidth=2)
    ax2.set_title('模型损失', fontsize=14, fontweight='bold')
    ax2.set_xlabel('训练轮次')
    ax2.set_ylabel('损失')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# 图片预处理函数 - 降低分辨率以提高速度
def preprocess_image(image, size=(64, 64)):
    """预处理图片，包括调整大小、归一化等"""
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image)


# 项目介绍页面
if page == "项目介绍":
    st.markdown('<div class="sub-header">🌿 欢迎，AI研究员！🌿</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        <div class="story-box">
        <h3>📖 任务背景</h3>
        <p>你已被野生动物保护组织招募为AI研究员，我们的保护区面临着巨大的挑战：</p>
        <ul>
        <li>红外相机每天拍摄数千张动物照片</li>
        <li>巡护员需要快速识别和分类这些动物</li>
        <li>我们需要追踪特定个体的健康状况</li>
        </ul>
        <p>你的任务是开发先进的AI系统，帮助巡护员更高效地保护这些珍贵的动物。</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="mission-card">
        <h3>🎯 你的任务</h3>
        <p><strong>第一阶段：</strong> 开发"保护区物种初筛系统"，能够区分不同动物物种</p>
        <p><strong>第二阶段：</strong> 升级为"动物个体识别追踪系统"，能够识别同一物种的不同个体</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="story-box">
        <h3>🌿 保护区内景</h3>
        <p>我们的自然保护区配备了先进的红外相机网络，能够24小时监测野生动物活动。</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="story-box">
        <h3>📊 数据收集</h3>
        <p>每天收集大量动物活动数据，需要AI系统帮助分析和识别。</p>
        </div>
        """, unsafe_allow_html=True)

    # 展示多种保护动物
    st.markdown('<div class="sub-header">🌍 保护区的珍贵居民 🌍</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div class="animal-card">
        <h3>🐼 大熊猫</h3>
        <p>黑白相间的毛色，圆滚滚的身体，爱吃竹子</p>
        <div class="animal-feature">
        <strong>特征：</strong>黑白毛色、圆脸、黑眼圈
        </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="animal-card">
        <h3>🦌 梅花鹿</h3>
        <p>身上有梅花状斑点，性情温顺</p>
        <div class="animal-feature">
        <strong>特征：</strong>梅花斑点、长腿、温顺
        </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="animal-card">
        <h3>🐯 东北虎</h3>
        <p>体型最大的猫科动物，威风凛凛</p>
        <div class="animal-feature">
        <strong>特征：</strong>条纹皮毛、强壮、独居
        </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="animal-card">
        <h3>🦅 金雕</h3>
        <p>猛禽之王，飞行速度极快</p>
        <div class="animal-feature">
        <strong>特征：</strong>钩状嘴、利爪、棕色羽毛
        </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="animal-card">
        <h3>🐒 金丝猴</h3>
        <p>拥有金色的毛发，活泼好动</p>
        <div class="animal-feature">
        <strong>特征：</strong>金色毛发、蓝脸、长尾巴
        </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="animal-card">
        <h3>🐘 亚洲象</h3>
        <p>陆地上最大的动物，智慧超群</p>
        <div class="animal-feature">
        <strong>特征：</strong>长鼻子、大耳朵、灰色皮肤
        </div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class="animal-card">
        <h3>🐆 雪豹</h3>
        <p>高山之王，毛色与雪地融为一体</p>
        <div class="animal-feature">
        <strong>特征：</strong>灰白毛色、长尾巴、斑点
        </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="animal-card">
        <h3>🦏 犀牛</h3>
        <p>体型庞大，鼻子上有角</p>
        <div class="animal-feature">
        <strong>特征：</strong>厚重皮肤、鼻角、体型大
        </div>
        </div>
        """, unsafe_allow_html=True)

    # 趣味知识
    st.markdown("""
    <div class="fun-fact">
    <h4>💡 你知道吗？</h4>
    <p>每只老虎的条纹都是独一无二的，就像人类的指纹一样！这让我们能够用AI技术来识别不同的老虎个体。</p>
    </div>
    """, unsafe_allow_html=True)

# 第一阶段：物种分类
elif page == "第一阶段：物种分类系统":
    st.markdown('<div class="sub-header">🔍 第一阶段：保护区物种初筛系统</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="story-box">
    <h3>📸 新任务：分类红外相机照片</h3>
    <p>保护区的红外相机刚刚传回了数百张新照片，巡护员需要你的帮助快速分类这些照片。</p>
    </div>
    """, unsafe_allow_html=True)

    # 使用三列布局
    col1, col2, col3 = st.columns([1, 1, 1])

    # 左侧：训练数据
    with col1:
        st.markdown('<div class="column-section">', unsafe_allow_html=True)
        st.subheader("📊 训练数据")

        st.markdown("""
        <div class="warning-box">
        <strong>注意：</strong> 请为每种动物上传图片！
        </div>
        """, unsafe_allow_html=True)

        # 类别设置
        num_classes = st.number_input("动物类别数量", min_value=2, max_value=10, value=3, step=1)

        # 为每个类别创建上传区域
        class_names = []
        for i in range(num_classes):
            # 生成默认类别名称
            default_name = f"动物{i + 1}"

            # 检查是否已有类别名称，如果有则使用现有的
            if i < len(st.session_state.class_names_phase1):
                default_name = st.session_state.class_names_phase1[i]

            class_name = st.text_input(f"类别 {i + 1} 名称", value=default_name, key=f"class_name_{i}")
            class_names.append(class_name)

            # 确保类别在class_images中
            if class_name not in st.session_state.class_images:
                st.session_state.class_images[class_name] = []

            # 文件上传器 - 修复上传问题
            uploader_key = f"class_uploader_{i}"
            uploaded_files = st.file_uploader(
                f"为 '{class_name}' 上传图片",
                type=['jpg', 'jpeg', 'png'],
                accept_multiple_files=True,
                key=uploader_key
            )

            # 检查是否有新上传的文件
            if uploaded_files and len(uploaded_files) > 0:
                # 检查是否与缓存中的文件不同
                cache_key = f"phase1_{class_name}"
                cached_files = st.session_state.uploaded_files_cache.get(cache_key, [])

                # 如果上传的文件与缓存不同，则更新
                if len(uploaded_files) != len(cached_files) or any(
                        uf.name != cf for uf, cf in zip(uploaded_files, cached_files)):
                    # 清空当前类别的图片，避免重复添加
                    st.session_state.class_images[class_name] = []

                    # 保存上传的图片
                    for uploaded_file in uploaded_files:
                        image = Image.open(uploaded_file).convert('RGB')
                        st.session_state.class_images[class_name].append(image)

                    # 更新缓存
                    st.session_state.uploaded_files_cache[cache_key] = [uf.name for uf in uploaded_files]

                    st.success(f"已为 '{class_name}' 上传 {len(uploaded_files)} 张图片")

                # 显示上传的图片样本
                if st.session_state.class_images[class_name]:
                    st.write(f"**{class_name}** 的图片样本:")
                    cols = st.columns(3)
                    for j, image in enumerate(st.session_state.class_images[class_name][:3]):
                        with cols[j % 3]:
                            st.image(image, caption=f"样本 {j + 1}", width=100)

            st.markdown("---")

        # 保存类别名称
        st.session_state.class_names_phase1 = class_names

        # 显示数据统计
        if st.session_state.class_images:
            st.subheader("📊 数据统计")
            total_images = 0
            for class_name in class_names:
                if class_name in st.session_state.class_images:
                    count = len(st.session_state.class_images[class_name])
                    st.write(f"- **{class_name}**: {count} 张图片")
                    total_images += count
            st.write(f"**总计**: {total_images} 张图片")

            if total_images < 3:
                st.warning("⚠️ 训练数据较少，建议每类至少上传3张图片以获得更好的模型效果")

        # 准备训练数据按钮
        if st.button("准备训练数据", type="primary"):
            if not st.session_state.class_images:
                st.error("请先上传训练数据！")
            else:
                with st.spinner("正在准备训练数据..."):
                    # 转换图片为PyTorch张量
                    images = []
                    labels = []

                    for class_idx, class_name in enumerate(class_names):
                        if class_name in st.session_state.class_images:
                            for image in st.session_state.class_images[class_name]:
                                try:
                                    # 使用改进的预处理
                                    image_tensor = preprocess_image(image)
                                    images.append(image_tensor)
                                    labels.append(class_idx)
                                except Exception as e:
                                    st.warning(f"无法处理 {class_name} 的图片: {str(e)}")

                    if len(images) < 2:
                        st.error("需要至少2张图片才能进行训练！")
                    else:
                        # 转换为PyTorch张量
                        images_tensor = torch.stack(images)
                        labels_tensor = torch.tensor(labels)

                        # 创建训练数据集
                        train_dataset = AnimalDataset(images_tensor, labels_tensor)
                        st.session_state.train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

                        st.success(f"✅ 训练数据准备完成！")
                        st.info(f"- 总图片数: {len(images)}")
                        st.info(f"- 图片尺寸: 64x64 (优化速度)")
                        st.info(f"- 批处理大小: 8")

        # 清除数据按钮
        if st.button("清除所有数据"):
            st.session_state.class_images = {}
            st.session_state.train_loader = None
            st.session_state.model = None
            st.session_state.training_history = None
            st.session_state.trained_phase1 = False
            st.session_state.uploaded_files_cache = {}
            st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

    # 中间：训练模型
    with col2:
        st.markdown('<div class="column-section">', unsafe_allow_html=True)
        st.subheader("🤖 训练模型")

        if not st.session_state.class_images:
            st.warning("请先上传训练数据！")
        else:
            st.markdown("""
            <div class="mission-card">
            <h4>快速PyTorch CNN模型</h4>
            <p>我们使用优化的卷积神经网络来获得更快的训练速度：</p>
            <ul>
            <li>3个卷积层提取特征</li>
            <li>更小的输入尺寸(64x64)</li>
            <li>减少参数数量</li>
            <li>轻量级网络结构</li>
            <li>优化训练参数</li>
            </ul>
            <p><strong>速度优化：</strong> 训练时间显著减少，适合课堂使用</p>
            </div>
            """, unsafe_allow_html=True)

            # 训练参数设置
            epochs = st.slider("训练轮次", 5, 50, 15)

            if st.button("开始训练模型", type="primary"):
                if st.session_state.train_loader is None:
                    st.error("请先准备训练数据！")
                else:
                    # 创建模型
                    num_classes = len(st.session_state.class_names_phase1)
                    model = FastAnimalCNN(num_classes).to(st.session_state.device)

                    # 设置优化器和损失函数
                    optimizer = optim.Adam(model.parameters(), lr=0.001)
                    criterion = nn.CrossEntropyLoss()

                    # 训练模型
                    with st.spinner("模型训练中，请稍候..."):
                        history = train_model(
                            model,
                            st.session_state.train_loader,
                            criterion,
                            optimizer,
                            epochs,
                            st.session_state.device
                        )

                    # 保存模型和训练历史
                    st.session_state.model = model
                    st.session_state.training_history = history
                    st.session_state.trained_phase1 = True

                    # 显示训练曲线
                    fig = plot_training_history(history)
                    st.pyplot(fig)

                    # 显示最终结果
                    final_train_acc = history['train_acc'][-1]
                    st.success(f"🎉 模型训练完成！最终训练准确率: {final_train_acc:.2f}%")

            # 提供模型下载
            if st.session_state.trained_phase1:
                st.subheader("📥 下载模型")
                if st.button("下载PyTorch模型"):
                    # 保存模型到字节流
                    buffer = io.BytesIO()
                    torch.save(st.session_state.model.state_dict(), buffer)
                    buffer.seek(0)

                    # 创建下载链接
                    b64 = base64.b64encode(buffer.read()).decode()
                    href = f'<a href="data:application/octet-stream;base64,{b64}" download="animal_classifier.pth">下载PyTorch模型文件</a>'
                    st.markdown(href, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # 右侧：测试模型
    with col3:
        st.markdown('<div class="column-section">', unsafe_allow_html=True)
        st.subheader("🔍 测试模型")

        if not st.session_state.trained_phase1:
            st.warning("请先训练模型！")
        else:
            # 单张图片预测
            st.subheader("单张图片预测")

            # 上传测试图片
            test_image = st.file_uploader(
                "上传测试图片",
                type=['jpg', 'jpeg', 'png'],
                key="test_uploader"
            )

            if test_image and st.session_state.model is not None:
                image = Image.open(test_image).convert('RGB')
                st.image(image, caption="测试图片", width=200)

                if st.button("识别动物", type="primary"):
                    # 预处理图片
                    model = st.session_state.model
                    model.eval()

                    # 使用相同的预处理
                    image_tensor = preprocess_image(image).unsqueeze(0)
                    image_tensor = image_tensor.to(st.session_state.device)

                    # 进行预测
                    with torch.no_grad():
                        output = model(image_tensor)
                        probabilities = torch.softmax(output, dim=1)
                        predicted_class = torch.argmax(probabilities, dim=1).item()
                        confidence = probabilities[0][predicted_class].item()

                    # 显示结果
                    predicted_name = st.session_state.class_names_phase1[predicted_class]

                    # 根据置信度显示不同的消息
                    if confidence > 0.8:
                        st.success(f"🔍 识别结果: **{predicted_name}**")
                    elif confidence > 0.6:
                        st.warning(f"🔍 识别结果: **{predicted_name}** (置信度较低)")
                    else:
                        st.error(f"🔍 识别结果: **{predicted_name}** (置信度很低)")

                    st.write(f"置信度: {confidence * 100:.2f}%")

                    # 显示所有类别的概率
                    st.subheader("所有类别概率:")
                    for i, class_name in enumerate(st.session_state.class_names_phase1):
                        prob = probabilities[0][i].item() * 100
                        color = "green" if i == predicted_class else "gray"
                        st.write(f"<span style='color:{color};'>{class_name}: {prob:.2f}%</span>",
                                 unsafe_allow_html=True)
                        st.progress(prob / 100)

            # 机器学习流程总结 - 调整顺序让学生自己总结
            st.markdown("---")
            st.subheader("🧠 总结机器学习流程")

            st.markdown("""
            <div class="learning-question">
            <h4>通过刚才的实践，你能总结出机器识别的基本流程吗？</h4>
            <p>请按照正确的顺序填写：</p>
            </div>
            """, unsafe_allow_html=True)

            # 让学生填写流程
            col1, col2, col3 = st.columns(3)

            with col1:
                step1 = st.text_input("第一步", value=st.session_state.learning_answers.get('step1', ''),
                                      placeholder="输入第一步流程")
                if step1:
                    st.session_state.learning_answers['step1'] = step1

            with col2:
                step2 = st.text_input("第二步", value=st.session_state.learning_answers.get('step2', ''),
                                      placeholder="输入第二步流程")
                if step2:
                    st.session_state.learning_answers['step2'] = step2

            with col3:
                step3 = st.text_input("第三步", value=st.session_state.learning_answers.get('step3', ''),
                                      placeholder="输入第三步流程")
                if step3:
                    st.session_state.learning_answers['step3'] = step3

            # 检查答案
            if st.button("检查我的答案"):
                correct_answers = ['输入数据', '训练模型', '测试模型']
                user_answers = [step1, step2, step3]

                if all(user_answers):
                    if (user_answers[0].strip() == '输入数据' and
                            user_answers[1].strip() == '训练模型' and
                            user_answers[2].strip() == '测试模型'):
                        st.success("🎉 完全正确！你成功总结了机器学习的基本流程！")
                    else:
                        st.warning("部分正确，请再思考一下流程顺序。正确答案是：输入数据 → 训练模型 → 测试模型")
                else:
                    st.error("请填写所有三个步骤")

        st.markdown('</div>', unsafe_allow_html=True)

# 第二阶段：个体识别
elif page == "第二阶段：个体识别系统":
    st.markdown('<div class="sub-header">🔬 第二阶段：动物个体识别追踪系统</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="story-box">
    <h3>🐼 新挑战：识别特定个体</h3>
    <p>现在我们发现保护区内每种动物都有多个个体，特别是大熊猫，我们需要知道"这是哪一只熊猫？"</p>
    <p>巡护员很难仅凭肉眼记住每一只熊猫的样子，尤其是在图片模糊、光线不好或只拍到局部的情况下。</p>
    </div>
    """, unsafe_allow_html=True)

    # 使用三列布局
    col1, col2, col3 = st.columns([1, 1, 1])

    # 左侧：训练数据
    with col1:
        st.markdown('<div class="column-section">', unsafe_allow_html=True)
        st.subheader("📊 个体数据")

        st.markdown("""
        <div class="warning-box">
        <br><strong>改进：</strong> 我们增强了模型对细微特征的识别能力，能更好地区分毛色、斑纹等个体特征。
        </div>
        """, unsafe_allow_html=True)

        # 类别设置
        num_individuals = st.number_input("个体数量", min_value=2, max_value=10, value=3, step=1)

        # 为每个个体创建上传区域
        individual_names = []
        for i in range(num_individuals):
            # 生成默认个体名称
            default_name = f"个体{i + 1}"

            # 检查是否已有个体名称，如果有则使用现有的
            if i < len(st.session_state.class_names_phase2):
                default_name = st.session_state.class_names_phase2[i]

            individual_name = st.text_input(f"个体 {i + 1} 名称", value=default_name, key=f"individual_name_{i}")
            individual_names.append(individual_name)

            # 确保个体在class_images_phase2中
            if individual_name not in st.session_state.class_images_phase2:
                st.session_state.class_images_phase2[individual_name] = []

            # 文件上传器 - 修复上传问题
            uploader_key = f"individual_uploader_{i}"
            uploaded_files = st.file_uploader(
                f"为 '{individual_name}' 上传图片",
                type=['jpg', 'jpeg', 'png'],
                accept_multiple_files=True,
                key=uploader_key
            )

            # 检查是否有新上传的文件
            if uploaded_files and len(uploaded_files) > 0:
                # 检查是否与缓存中的文件不同
                cache_key = f"phase2_{individual_name}"
                cached_files = st.session_state.uploaded_files_cache.get(cache_key, [])

                # 如果上传的文件与缓存不同，则更新
                if len(uploaded_files) != len(cached_files) or any(
                        uf.name != cf for uf, cf in zip(uploaded_files, cached_files)):
                    # 清空当前个体的图片，避免重复添加
                    st.session_state.class_images_phase2[individual_name] = []

                    # 保存上传的图片
                    for uploaded_file in uploaded_files:
                        image = Image.open(uploaded_file).convert('RGB')
                        st.session_state.class_images_phase2[individual_name].append(image)

                    # 更新缓存
                    st.session_state.uploaded_files_cache[cache_key] = [uf.name for uf in uploaded_files]

                    st.success(f"已为 '{individual_name}' 上传 {len(uploaded_files)} 张图片")

                # 显示上传的图片样本
                if st.session_state.class_images_phase2[individual_name]:
                    st.write(f"**{individual_name}** 的图片样本:")
                    cols = st.columns(3)
                    for j, image in enumerate(st.session_state.class_images_phase2[individual_name][:3]):
                        with cols[j % 3]:
                            st.image(image, caption=f"样本 {j + 1}", width=100)

            st.markdown("---")

        # 保存个体名称
        st.session_state.class_names_phase2 = individual_names

        # 显示数据统计
        if st.session_state.class_images_phase2:
            st.subheader("📊 数据统计")
            total_images = 0
            for individual_name in individual_names:
                if individual_name in st.session_state.class_images_phase2:
                    count = len(st.session_state.class_images_phase2[individual_name])
                    st.write(f"- **{individual_name}**: {count} 张图片")
                    total_images += count
            st.write(f"**总计**: {total_images} 张图片")

            if total_images < 3:
                st.warning("⚠️ 个体识别需要更多数据！建议每个个体至少上传3张不同角度的图片")

        # 准备训练数据按钮
        if st.button("准备训练数据", type="primary", key="phase2_preprocess"):
            if not st.session_state.class_images_phase2:
                st.error("请先上传个体数据！")
            else:
                with st.spinner("正在准备训练数据..."):
                    # 转换图片为PyTorch张量
                    images = []
                    labels = []

                    for individual_idx, individual_name in enumerate(individual_names):
                        if individual_name in st.session_state.class_images_phase2:
                            for image in st.session_state.class_images_phase2[individual_name]:
                                try:
                                    # 使用改进的预处理
                                    image_tensor = preprocess_image(image)
                                    images.append(image_tensor)
                                    labels.append(individual_idx)
                                except Exception as e:
                                    st.warning(f"无法处理 {individual_name} 的图片: {str(e)}")

                    if len(images) < 2:
                        st.error("需要至少2张图片才能进行训练！")
                    else:
                        # 转换为PyTorch张量
                        images_tensor = torch.stack(images)
                        labels_tensor = torch.tensor(labels)

                        # 创建训练数据集
                        train_dataset = AnimalDataset(images_tensor, labels_tensor)
                        st.session_state.train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

                        st.success(f"✅ 训练数据准备完成！")
                        st.info(f"- 总图片数: {len(images)}")
                        st.info(f"- 图片尺寸: 64x64")
                        st.info(f"- 批处理大小: 8")

        # 清除数据按钮
        if st.button("清除所有数据", key="phase2_clear"):
            st.session_state.class_images_phase2 = {}
            st.session_state.train_loader = None
            st.session_state.model = None
            st.session_state.training_history = None
            st.session_state.trained_phase2 = False
            st.session_state.uploaded_files_cache = {}
            st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

    # 中间：训练模型
    with col2:
        st.markdown('<div class="column-section">', unsafe_allow_html=True)
        st.subheader("🤖 训练模型")

        if not st.session_state.class_images_phase2:
            st.warning("请先上传个体数据！")
        else:
            st.markdown("""
            <div class="mission-card">
            <h4>快速个体识别模型</h4>
            <p>我们使用优化的网络结构来提高个体识别速度：</p>
            <ul>
            <li>轻量级卷积网络(3层)</li>
            <li>更小的输入尺寸(64x64)</li>
            <li>减少参数数量</li>
            <li>优化训练参数</li>
            </ul>
            <p><strong>速度优势：</strong> 训练时间大幅减少，适合课堂实践</p>
            </div>
            """, unsafe_allow_html=True)

            # 训练参数设置
            epochs = st.slider("训练轮次", 5, 50, 20, key="phase2_epochs")

            if st.button("开始训练模型", type="primary", key="phase2_train"):
                if st.session_state.train_loader is None:
                    st.error("请先准备训练数据！")
                else:
                    # 创建模型
                    num_classes = len(st.session_state.class_names_phase2)
                    model = FastAnimalCNN(num_classes).to(st.session_state.device)

                    # 设置优化器和损失函数
                    optimizer = optim.Adam(model.parameters(), lr=0.001)
                    criterion = nn.CrossEntropyLoss()

                    # 训练模型
                    with st.spinner("个体识别模型训练中..."):
                        history = train_model(
                            model,
                            st.session_state.train_loader,
                            criterion,
                            optimizer,
                            epochs,
                            st.session_state.device
                        )

                    # 保存模型和训练历史
                    st.session_state.model = model
                    st.session_state.training_history = history
                    st.session_state.trained_phase2 = True

                    # 显示训练曲线
                    fig = plot_training_history(history)
                    st.pyplot(fig)

                    # 显示最终结果
                    final_train_acc = history['train_acc'][-1]
                    st.success(f"🎉 个体识别模型训练完成！最终训练准确率: {final_train_acc:.2f}%")

            # 提供模型下载
            if st.session_state.trained_phase2:
                st.subheader("📥 下载模型")
                if st.button("下载PyTorch模型", key="phase2_download"):
                    # 保存模型到字节流
                    buffer = io.BytesIO()
                    torch.save(st.session_state.model.state_dict(), buffer)
                    buffer.seek(0)

                    # 创建下载链接
                    b64 = base64.b64encode(buffer.read()).decode()
                    href = f'<a href="data:application/octet-stream;base64,{b64}" download="individual_recognizer.pth">下载PyTorch模型文件</a>'
                    st.markdown(href, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # 右侧：测试模型
    with col3:
        st.markdown('<div class="column-section">', unsafe_allow_html=True)
        st.subheader("🔍 测试模型")

        if not st.session_state.trained_phase2:
            st.warning("请先训练个体识别模型！")
        else:
            # 单张图片预测
            st.subheader("单张图片预测")

            # 上传测试图片
            test_image = st.file_uploader(
                "上传测试图片",
                type=['jpg', 'jpeg', 'png'],
                key="phase2_test_uploader"
            )

            if test_image and st.session_state.model is not None:
                image = Image.open(test_image).convert('RGB')
                st.image(image, caption="测试图片", width=200)

                if st.button("识别个体", type="primary", key="phase2_predict"):
                    # 预处理图片
                    model = st.session_state.model
                    model.eval()

                    # 使用相同的预处理
                    image_tensor = preprocess_image(image).unsqueeze(0)
                    image_tensor = image_tensor.to(st.session_state.device)

                    # 进行预测
                    with torch.no_grad():
                        output = model(image_tensor)
                        probabilities = torch.softmax(output, dim=1)
                        predicted_class = torch.argmax(probabilities, dim=1).item()
                        confidence = probabilities[0][predicted_class].item()

                    # 显示结果
                    predicted_name = st.session_state.class_names_phase2[predicted_class]

                    # 根据置信度显示不同的消息
                    if confidence > 0.85:
                        st.success(f"🔍 识别结果: **{predicted_name}**")
                        st.balloons()
                    elif confidence > 0.7:
                        st.warning(f"🔍 识别结果: **{predicted_name}** (中等置信度)")
                    else:
                        st.error(f"🔍 识别结果: **{predicted_name}** (低置信度，建议检查图片质量)")

                    st.write(f"置信度: {confidence * 100:.2f}%")

                    # 显示所有个体的概率
                    st.subheader("所有个体概率:")
                    for i, individual_name in enumerate(st.session_state.class_names_phase2):
                        prob = probabilities[0][i].item() * 100
                        color = "green" if i == predicted_class else "gray"
                        st.write(f"<span style='color:{color};'>{individual_name}: {prob:.2f}%</span>",
                                 unsafe_allow_html=True)
                        st.progress(prob / 100)

        st.markdown('</div>', unsafe_allow_html=True)

# 学习单页面
elif page == "学习单":
    st.markdown('<div class="sub-header">📚 《机器学习之动物保护》学习单</div>', unsafe_allow_html=True)

    # 学习单内容
    st.markdown("""
    <div class="learning-sheet">
    <h2>《机器学习之动物保护》学习单</h2>
    </div>
    """, unsafe_allow_html=True)

    # 一、学习目标
    st.markdown("""
    <div class="learning-question">
    <h3>🎯 一、学习目标</h3>
    <p>1. 理解机器学习概念，掌握机器学习的基本流程</p>
    <p>2. 用"动物保护AI识别系统"完成物种分类、个体识别模型，总结数据对模型效果的重要性</p>
    <p>3. 感知 AI 对动物保护的帮助，能举例说明生活中的机器学习应用</p>
    </div>
    """, unsafe_allow_html=True)

    # 二、课堂实践
    st.markdown("""
    <div class="learning-question">
    <h3>🔬 二、课堂实践</h3>
    <p>巡护员在野外需快速区分珍稀动物，减少人工识别时间，避免误判。巡护员如何区分这些动物呢？</p>
    <p>动物的特征不同：请列举以下动物的特点</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        panda_features = st.text_area("大熊猫特征",
                                      value=st.session_state.learning_answers.get('panda_features', ''),
                                      placeholder="黑白相间的毛色，圆滚滚的身体，爱吃竹子...",
                                      height=100)
        if panda_features:
            st.session_state.learning_answers['panda_features'] = panda_features

    with col2:
        tiger_features = st.text_area("老虎特征",
                                      value=st.session_state.learning_answers.get('tiger_features', ''),
                                      placeholder="有条纹皮毛，体型强壮，是独居动物...",
                                      height=100)
        if tiger_features:
            st.session_state.learning_answers['tiger_features'] = tiger_features

    with col3:
        monkey_features = st.text_area("金丝猴特征",
                                       value=st.session_state.learning_answers.get('monkey_features', ''),
                                       placeholder="金色毛发，蓝色面孔，长尾巴...",
                                       height=100)
        if monkey_features:
            st.session_state.learning_answers['monkey_features'] = monkey_features

    # 机器学习概念
    st.markdown("""
    <div class="learning-question">
    <h3>🤖 1. 机器学习的概念</h3>
    <p>机器学习是让机器________________，获得知识与技能，从而感知世界、认识世界的技术。</p>
    </div>
    """, unsafe_allow_html=True)

    ml_concept = st.text_input("填写机器学习概念",
                               value=st.session_state.learning_answers.get('ml_concept', ''),
                               placeholder="")
    if ml_concept:
        st.session_state.learning_answers['ml_concept'] = ml_concept

    # 活动1：探索物种分类模型
    st.markdown("""
    <div class="step-box">
    <h3>🔍 活动1：探索物种分类模型</h3>
    <h4>1. 实践操作</h4>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        panda_count = st.number_input("大熊猫图片数量", min_value=0, max_value=100, value=0)
        st.session_state.learning_answers['panda_count'] = panda_count

    with col2:
        tiger_count = st.number_input("老虎图片数量", min_value=0, max_value=100, value=0)
        st.session_state.learning_answers['tiger_count'] = tiger_count

    with col3:
        monkey_count = st.number_input("金丝猴图片数量", min_value=0, max_value=100, value=0)
        st.session_state.learning_answers['monkey_count'] = monkey_count

    test_count = st.number_input("测试图片数量", min_value=0, max_value=50, value=0)
    accuracy = st.slider("识别准确率 (%)", 0, 100, 0)

    st.session_state.learning_answers['test_count'] = test_count
    st.session_state.learning_answers['accuracy'] = accuracy

    # 机器学习流程总结
    st.markdown("""
    <div class="learning-question">
    <h4>2. 总结流程：</h4>
    <p>通过活动1的实践探索，填写<strong>机器识别的基本流程</strong>：</p>
    </div>
    """, unsafe_allow_html=True)

    # 使用之前填写的流程答案
    process_step1 = st.text_input("第一步流程",
                                  value=st.session_state.learning_answers.get('step1', ''),
                                  key="sheet_step1")
    process_step2 = st.text_input("第二步流程",
                                  value=st.session_state.learning_answers.get('step2', ''),
                                  key="sheet_step2")
    process_step3 = st.text_input("第三步流程",
                                  value=st.session_state.learning_answers.get('step3', ''),
                                  key="sheet_step3")

    # 活动2：探索个体识别模型
    st.markdown("""
    <div class="step-box">
    <h3>🔬 活动2：探索个体识别模型</h3>
    <p>新任务发布："升级AI系统，实现对不同个体的行动轨迹监测，这是精准保护的关键。"</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="learning-question">
    <h4>任务1：</h4>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        individual1_count = st.number_input("个体1图片数量", min_value=0, max_value=100, value=0)

    with col2:
        individual2_count = st.number_input("个体2图片数量", min_value=0, max_value=100, value=0)

    with col3:
        individual3_count = st.number_input("个体3图片数量", min_value=0, max_value=100, value=0)

    individual_test_count = st.number_input("个体测试图片数量", min_value=0, max_value=50, value=0)
    individual_accuracy = st.slider("个体识别准确率 (%)", 0, 100, 0)

    # 个体识别结果
    st.markdown("""
    <div class="learning-question">
    <p>结果：</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        can_recognize = st.radio("能否识别个体", ["能识别个体", "不能识别个体"])

    with col2:
        reason = st.text_area("原因分析",
                              placeholder="分析为什么能或不能识别个体...",
                              height=100)

    # 任务2：分组比较
    st.markdown("""
    <div class="learning-question">
    <h4>任务2：分组比较</h4>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**第1组**")
        group1_accuracy = st.slider("第1组准确率", 0, 100, 0)
        group1_features = st.text_area("第1组图片特点", placeholder="描述图片特点...", height=80)

    with col2:
        st.markdown("**第2组**")
        group2_accuracy = st.slider("第2组准确率", 0, 100, 0)
        group2_features = st.text_area("第2组图片特点", placeholder="描述图片特点...", height=80)

    better_group = st.radio("哪组模型更准确", ["第1组", "第2组"])
    better_reason = st.text_area("更准确的原因", placeholder="分析为什么这组更准确...")

    # 总结
    st.markdown("""
    <div class="learning-question">
    <h3>📝 总结</h3>
    <p>______________是人工智能的核心要素，就像人类需要通过学习积累知识一样，智能系统也需要通过大量______________来训练自己，通过分析______________中的规律，逐渐学会完成特定任务。______________越多样、质量越高，人工智能的学习效果就越好。</p>
    </div>
    """, unsafe_allow_html=True)

    summary_answers = st.text_area("填写总结",
                                   placeholder="",
                                   height=100)

    st.success("🎉 恭喜你成为优秀的野生动物保护AI研究员！")

    # 拓展学习
    st.markdown("""
    <div class="learning-question">
    <h3>💡 三、拓展学习</h3>
    <p>思考：机器学习在生活中的应用有哪些？</p>
    </div>
    """, unsafe_allow_html=True)

    ml_applications = st.text_area("机器学习应用举例",
                                   placeholder=" ",
                                   height=100)

    # 学习评价
    st.markdown("""
    <div class="learning-question">
    <h3>⭐ 四、学习评价</h3>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**1. 经过本课的学习，你有哪些收获呢？**")
    harvest = st.text_area("学习收获", placeholder="写下你的收获...", height=100)

    st.markdown("**2. 你认为以下描述是否正确？**")

    col1, col2, col3 = st.columns([3, 1, 1])

    with col1:
        st.write("**描述**")
        st.write("(1) 机器学习是人工智能领域的核心技术")
        st.write("(2) 机器学习技术可以使AI获得归纳推理和决策能力")
        st.write("(3) 机器学习技术可以解决人工智能领域的所有问题")

    with col2:
        st.write("**是**")
        q1_correct = st.checkbox(" ", key="q1")
        q2_correct = st.checkbox(" ", key="q2")
        q3_correct = st.checkbox(" ", key="q3")

    with col3:
        st.write("**否**")
        q1_wrong = st.checkbox(" ", key="q1_w")
        q2_wrong = st.checkbox(" ", key="q2_w")
        q3_wrong = st.checkbox(" ", key="q3_w")

    # 小组活动评价
    st.markdown("**3. 小组活动评价**")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**(1) 我觉得认识了机器学习的概念及基本流程，了解数据在人工智能领域的重要性**")
        rating1 = st.slider("评分", 1, 5, 3, key="rating1")
        st.write("⭐" * rating1)

    with col2:
        st.write("**(2) 在课堂互动环节中，我有积极地参与到课堂的互动中来**")
        rating2 = st.slider("评分", 1, 5, 3, key="rating2")
        st.write("⭐" * rating2)

    # 课堂总结
    st.markdown("""
    <div class="learning-question">
    <h3>📚 五、课堂总结（梳理与反思）</h3>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**1. 这节课你学到的核心知识点：**")
    key_points = st.text_area("核心知识点", placeholder="", height=100)

    st.markdown("**2. 关于机器学习，你还有哪些疑问？**")
    questions = st.text_area("疑问与思考", placeholder="写下你的疑问...", height=100)

    # 保存学习单答案
    if st.button("保存学习单答案"):
        st.session_state.learning_answers.update({
            'panda_features': panda_features,
            'tiger_features': tiger_features,
            'monkey_features': monkey_features,
            'ml_concept': ml_concept,
            'panda_count': panda_count,
            'tiger_count': tiger_count,
            'monkey_count': monkey_count,
            'test_count': test_count,
            'accuracy': accuracy,
            'individual1_count': individual1_count,
            'individual2_count': individual2_count,
            'individual3_count': individual3_count,
            'individual_test_count': individual_test_count,
            'individual_accuracy': individual_accuracy,
            'can_recognize': can_recognize,
            'reason': reason,
            'group1_accuracy': group1_accuracy,
            'group1_features': group1_features,
            'group2_accuracy': group2_accuracy,
            'group2_features': group2_features,
            'better_group': better_group,
            'better_reason': better_reason,
            'summary_answers': summary_answers,
            'ml_applications': ml_applications,
            'harvest': harvest,
            'key_points': key_points,
            'questions': questions
        })
        st.success("学习单答案已保存！")

# AI研究员证书页面
elif page == "AI研究员证书":
    st.markdown('<div class="sub-header">🏆 AI研究员成就证书</div>', unsafe_allow_html=True)

    # 创建证书
    st.markdown("""
    <div style="border: 10px solid #FFD700; padding: 40px; text-align: center; background: linear-gradient(135deg, #E3F2FD, #BBDEFB); border-radius: 20px;">
    <h1 style="color: #1565C0; font-size: 3rem; margin-bottom: 20px;">🎓 AI研究员证书</h1>
    <p style="font-size: 1.5rem; color: #333; margin-bottom: 30px;">授予优秀的野生动物保护AI研究员</p>

    <div style="background: white; padding: 30px; border-radius: 15px; margin: 20px 0; border: 2px solid #64B5F6;">
    <h2 style="color: #1976D2; font-size: 2.5rem; margin-bottom: 10px;">动物保护AI专家</h2>
    <p style="font-size: 1.3rem; color: #555; margin-bottom: 20px;">成功完成动物识别AI系统开发</p>

    <div style="display: flex; justify-content: space-around; margin: 30px 0;">
    <div>
    <h3 style="color: #388E3C;">机器学习的概念</h3>
    <p style="font-size: 1.2rem;">🎯 熟练掌握</p>
    </div>
    <div>
    <h3 style="color: #F57C00;">机器学习的基本流程</h3>
    <p style="font-size: 1.2rem;">🎯 实践应用</p>
    </div>
    <div>
    <h3 style="color: #7B1FA2;">数据</h3>
    <p style="font-size: 1.2rem;">📊 深度理解</p>
    </div>
    </div>

    <p style="font-size: 1.1rem; color: #666; font-style: italic;">
    "运用PyTorch和AI技术为野生动物保护做出重要贡献"
    </p>
    </div>

    <p style="font-size: 1.2rem; color: #333; margin-top: 20px;">
    <strong>颁发机构：</strong>野生动物保护组织AI研究部
    </p>

    <p style="font-size: 1.1rem; color: #666;">
    日期：2025年 • 荣誉证书
    </p>
    </div>
    """, unsafe_allow_html=True)

    # 技能总结
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "🐾 野生动物保护组织 AI 研究部 • 用科技守护生命 🐾"
        "</div>",
        unsafe_allow_html=True
    )
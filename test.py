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

# 页面配置 - 修改为更适合的布局
st.set_page_config(
    page_title="动物保护 AI 识别系统",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="collapsed"  # 侧边栏默认折叠
)

# 自定义CSS美化界面 - 更适合初中生
st.markdown("""\
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
/* 美化侧边栏 - 改为浅绿色 */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #90EE90 0%, #98FB98 100%);
}
[data-testid="stSidebar"] .sidebar-content {
    color: #2E8B57;
}
[data-testid="stSidebar"] .stRadio > div {
    color: #2E8B57;
}
[data-testid="stSidebar"] label {
    color: #2E8B57 !important;
    font-weight: bold;
}
/* 文件上传器样式 */
.uploadedFile {
    border: 2px dashed #4CAF50 !important;
    background-color: #F0FFF0 !important;
}
/* 紧凑布局样式 */
.compact-section {
    margin-bottom: 1rem;
    padding: 1rem;
    background: #f9f9f9;
    border-radius: 8px;
}
/* 新增：两列布局样式 */
.two-column-layout {
    display: flex;
    gap: 20px;
    margin-bottom: 20px;
}
.column {
    flex: 1;
}
/* 上传提示样式 */
.upload-hint {
    background: linear-gradient(135deg, #E8F5E9, #C8E6C9);
    padding: 1rem;
    border-radius: 10px;
    margin: 0.5rem 0;
    border-left: 4px solid #4CAF50;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)

# 应用标题
st.markdown('<div class="main-header">🐾 动物保护 AI 识别系统 🐾</div>', unsafe_allow_html=True)

# 侧边栏导航 - 美化
st.sidebar.title("🌿 导航")
page = st.sidebar.radio("选择任务阶段:",
                        ["项目介绍",
                         "学习单",
                         "第一阶段：物种分类系统",
                         "第二阶段：个体识别系统",
                         "AI研究员证书"])

# 初始化session state - 确保数据持久化
if 'model_phase1' not in st.session_state:
    st.session_state.model_phase1 = None
if 'model_phase2' not in st.session_state:
    st.session_state.model_phase2 = None
if 'training_history_phase1' not in st.session_state:
    st.session_state.training_history_phase1 = None
if 'training_history_phase2' not in st.session_state:
    st.session_state.training_history_phase2 = None
if 'class_names_phase1' not in st.session_state:
    st.session_state.class_names_phase1 = []
if 'class_names_phase2' not in st.session_state:
    st.session_state.class_names_phase2 = []
if 'device' not in st.session_state:
    st.session_state.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if 'train_loader_phase1' not in st.session_state:
    st.session_state.train_loader_phase1 = None
if 'train_loader_phase2' not in st.session_state:
    st.session_state.train_loader_phase2 = None
if 'trained_phase1' not in st.session_state:
    st.session_state.trained_phase1 = False
if 'trained_phase2' not in st.session_state:
    st.session_state.trained_phase2 = False
if 'class_images_phase1' not in st.session_state:
    st.session_state.class_images_phase1 = {}
if 'class_images_phase2' not in st.session_state:
    st.session_state.class_images_phase2 = {}
if 'uploaded_files_cache' not in st.session_state:
    st.session_state.uploaded_files_cache = {}
if 'learning_answers' not in st.session_state:
    st.session_state.learning_answers = {}
# 添加重置计数器
if 'reset_counter_phase1' not in st.session_state:
    st.session_state.reset_counter_phase1 = 0
if 'reset_counter_phase2' not in st.session_state:
    st.session_state.reset_counter_phase2 = 0
# 添加步骤标题状态
if 'step1_title' not in st.session_state:
    st.session_state.step1_title = "第一步"
if 'step2_title' not in st.session_state:
    st.session_state.step2_title = "第二步"
if 'step3_title' not in st.session_state:
    st.session_state.step3_title = "第三步"


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


# 定义改进的CNN模型 - 增强特征提取能力
class ImprovedAnimalCNN(nn.Module):
    def __init__(self, num_classes):
        super(ImprovedAnimalCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )

        # 动态计算全连接层输入尺寸
        self.fc_input_size = self._get_fc_input_size()

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.fc_input_size, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
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


# 改进的训练函数 - 优化训练速度和准确率
def train_model(model, train_loader, criterion, optimizer, epochs, device, phase=1):
    train_losses = []
    train_accs = []

    # 创建进度条和状态文本
    progress_bar = st.progress(0)
    status_text = st.empty()
    time_placeholder = st.empty()

    # 学习率调度器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

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

        # 更新学习率
        scheduler.step()

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # 计算预计剩余时间
        elapsed_time = time.time() - start_time
        time_per_epoch = elapsed_time / (epoch + 1)
        remaining_time = time_per_epoch * (epochs - epoch - 1)

        phase_name = "物种分类" if phase == 1 else "个体识别"
        status_text.text(f"{phase_name}模型训练中... 第 {epoch + 1}/{epochs} 轮")
        time_placeholder.markdown(f"""\
<div class="progress-info">
<strong>进度:</strong> {epoch + 1}/{epochs} 轮<br>
<strong>当前准确率:</strong> {train_acc:.2f}%<br>
<strong>当前损失:</strong> {train_loss:.4f}<br>
<strong>预计剩余时间:</strong> {remaining_time:.1f}秒
</div>
""", unsafe_allow_html=True)

    # 清除状态文本
    status_text.text("训练完成！")
    progress_bar.empty()
    time_placeholder.empty()

    total_time = time.time() - start_time
    phase_name = "物种分类" if phase == 1 else "个体识别"
    st.success(f"{phase_name}模型训练完成！总耗时: {total_time:.1f}秒")

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


# 优化的图片预处理函数 - 提高处理速度
def preprocess_image(image, size=(64, 64)):
    """预处理图片，包括调整大小、归一化等"""
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image)


# 改进的上传处理函数 - 解决重命名后上传问题
def handle_file_upload(uploaded_files, class_name, idx, phase=1):
    """处理文件上传，优化上传速度"""
    if uploaded_files is not None and len(uploaded_files) > 0:
        # 使用索引而不是类别名称作为缓存键，避免类别名称修改导致的问题
        cache_key = f"phase{phase}_{idx}"

        # 获取当前实际类别名称
        if phase == 1:
            if idx < len(st.session_state.class_names_phase1):
                actual_class_name = st.session_state.class_names_phase1[idx]
            else:
                actual_class_name = class_name
        else:
            if idx < len(st.session_state.class_names_phase2):
                actual_class_name = st.session_state.class_names_phase2[idx]
            else:
                actual_class_name = class_name

        # 检查是否需要更新
        cached_files = st.session_state.uploaded_files_cache.get(cache_key, [])
        current_files = [uf.name for uf in uploaded_files]

        # 如果上传的文件与缓存不同，或者缓存为空，或者类别名称已更改，则更新
        if (current_files != cached_files or not cached_files or
                actual_class_name not in (
                st.session_state.class_images_phase1 if phase == 1 else st.session_state.class_images_phase2)):

            # 使用进度条显示上传进度
            progress_bar = st.progress(0)
            status_text = st.empty()

            # 清空当前类别的图片，避免重复添加
            if phase == 1:
                st.session_state.class_images_phase1[actual_class_name] = []
            else:
                st.session_state.class_images_phase2[actual_class_name] = []

            # 批量处理图片上传
            total_files = len(uploaded_files)
            for i, uploaded_file in enumerate(uploaded_files):
                try:
                    # 快速打开图片并转换为RGB
                    image = Image.open(uploaded_file).convert('RGB')
                    if phase == 1:
                        st.session_state.class_images_phase1[actual_class_name].append(image)
                    else:
                        st.session_state.class_images_phase2[actual_class_name].append(image)

                    # 更新进度
                    progress = (i + 1) / total_files
                    progress_bar.progress(progress)
                    status_text.text(f"处理图片 {i + 1}/{total_files}")

                except Exception as e:
                    st.warning(f"无法处理文件 {uploaded_file.name}: {str(e)}")

            # 清除进度条
            progress_bar.empty()
            status_text.empty()

            # 更新缓存
            st.session_state.uploaded_files_cache[cache_key] = current_files

        return True
    return False


# 项目介绍页面
if page == "项目介绍":
    st.markdown('<div class="sub-header">🌿 欢迎，AI研究员！ 🌿</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""\
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

        st.markdown("""\
<div class="mission-card">
<h3>🎯 你的任务</h3>
<p><strong>第一阶段：</strong> 开发"保护区物种初筛系统"，能够区分不同动物物种</p>
<p><strong>第二阶段：</strong> 升级为"动物个体识别追踪系统"，能够识别同一物种的不同个体</p>
</div>
""", unsafe_allow_html=True)

    with col2:
        st.markdown("""\
<div class="story-box">
<h3>🌿 保护区内景</h3>
<p>我们的自然保护区配备了先进的红外相机网络，能够24小时监测野生动物活动。</p>
</div>
""", unsafe_allow_html=True)

        st.markdown("""\
<div class="story-box">
<h3>📊 数据收集</h3>
<p>每天收集大量动物活动数据，需要AI系统帮助分析和识别。</p>
</div>
""", unsafe_allow_html=True)

    # 展示多种保护动物 - 恢复8种动物
    st.markdown('<div class="sub-header">🌍 保护区的珍贵居民 🌍</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""\
<div class="animal-card">
<h3>🐼 大熊猫</h3>
<p>黑白相间的毛色，圆滚滚的身体，爱吃竹子</p>
<div class="animal-feature">
<strong>特征：</strong>黑白毛色、圆脸、黑眼圈
</div>
</div>
""", unsafe_allow_html=True)

        st.markdown("""\
<div class="animal-card">
<h3>🦌 梅花鹿</h3>
<p>身上有梅花状斑点，性情温顺</p>
<div class="animal-feature">
<strong>特征：</strong>梅花斑点、长腿、温顺
</div>
</div>
""", unsafe_allow_html=True)

    with col2:
        st.markdown("""\
<div class="animal-card">
<h3>🐯 东北虎</h3>
<p>体型最大的猫科动物，威风凛凛</p>
<div class="animal-feature">
<strong>特征：</strong>条纹皮毛、强壮、独居
</div>
</div>
""", unsafe_allow_html=True)

        st.markdown("""\
<div class="animal-card">
<h3>🦅 金雕</h3>
<p>猛禽之王，飞行速度极快</p>
<div class="animal-feature">
<strong>特征：</strong>钩状嘴、利爪、棕色羽毛
</div>
</div>
""", unsafe_allow_html=True)

    with col3:
        st.markdown("""\
<div class="animal-card">
<h3>🐒 金丝猴</h3>
<p>拥有金色的毛发，活泼好动</p>
<div class="animal-feature">
<strong>特征：</strong>金色毛发、蓝脸、长尾巴
</div>
</div>
""", unsafe_allow_html=True)

        st.markdown("""\
<div class="animal-card">
<h3>🐘 亚洲象</h3>
<p>陆地上最大的动物，智慧超群</p>
<div class="animal-feature">
<strong>特征：</strong>长鼻子、大耳朵、灰色皮肤
</div>
</div>
""", unsafe_allow_html=True)

    with col4:
        st.markdown("""\
<div class="animal-card">
<h3>🐆 雪豹</h3>
<p>高山之王，毛色与雪地融为一体</p>
<div class="animal-feature">
<strong>特征：</strong>灰白毛色、长尾巴、斑点
</div>
</div>
""", unsafe_allow_html=True)

        st.markdown("""\
<div class="animal-card">
<h3>🦏 犀牛</h3>
<p>体型庞大，鼻子上有角</p>
<div class="animal-feature">
<strong>特征：</strong>厚重皮肤、鼻角、体型大
</div>
</div>
""", unsafe_allow_html=True)

    # 趣味知识
    st.markdown("""\
<div class="fun-fact">
<h4>💡 你知道吗？</h4>
<p>每只老虎的条纹都是独一无二的，就像人类的指纹一样！这让我们能够用AI技术来识别不同的老虎个体。</p>
</div>
""", unsafe_allow_html=True)

# 第一阶段：物种分类 - 改为上下结构
elif page == "第一阶段：物种分类系统":
    st.markdown('<div class="sub-header">🔍 第一阶段：保护区物种初筛系统</div>', unsafe_allow_html=True)

    st.markdown("""\
<div class="story-box">
<h3>📸 新任务：分类红外相机照片</h3>
<p>保护区的红外相机刚刚传回了数百张新照片，巡护员需要你的帮助快速分类这些照片。</p>
</div>
""", unsafe_allow_html=True)

    # 第一部分：训练数据
    st.markdown(f'<div class="sub-header">📊 {st.session_state.step1_title}</div>', unsafe_allow_html=True)

    st.markdown("""\
<div class="warning-box">
<strong>注意：</strong> 请为每种动物上传至少5张图片！
</div>
""", unsafe_allow_html=True)

    # 类别设置 - 修改默认值为2
    num_classes = st.number_input("动物类别数量", min_value=2, max_value=10, value=2, step=1)

    # 使用两列布局显示类别
    class_names = []
    for i in range(num_classes):
        # 生成默认类别名称
        default_name = f"动物{i + 1}"

        # 检查是否已有类别名称，如果有则使用现有的
        if i < len(st.session_state.class_names_phase1):
            default_name = st.session_state.class_names_phase1[i]

        # 每行显示两个类别
        if i % 2 == 0:  # 偶数索引创建新行
            cols = st.columns(2)

        with cols[i % 2]:  # 在当前行的对应列中显示
            with st.container():
                st.markdown(f"**类别 {i + 1}**")
                class_name = st.text_input(f"类别名称", value=default_name, key=f"class_name_{i}")
                class_names.append(class_name)

                # 确保类别在class_images_phase1中
                if class_name not in st.session_state.class_images_phase1:
                    st.session_state.class_images_phase1[class_name] = []

                # 文件上传器 - 使用包含重置计数器的key
                uploader_key = f"class_uploader_{i}_{st.session_state.reset_counter_phase1}"

                # 美化的上传提示
                st.markdown("""\
<div class="upload-hint">
📸 <strong>上传图片：</strong> 请点击下方区域选择图片或将图片拖拽到此处
</div>
""", unsafe_allow_html=True)

                uploaded_files = st.file_uploader(
                    f"为 '{class_name}' 上传图片",
                    type=['jpg', 'jpeg', 'png'],
                    accept_multiple_files=True,
                    key=uploader_key,
                    help="支持JPG、JPEG、PNG格式，最多可上传200张图片"
                )

                # 使用改进的上传处理
                if handle_file_upload(uploaded_files, class_name, i, phase=1):
                    st.success(f"已为 '{class_name}' 上传 {len(uploaded_files)} 张图片")

                # 显示上传的图片样本（不显示附件列表）
                if st.session_state.class_images_phase1.get(class_name):
                    images = st.session_state.class_images_phase1[class_name]
                    if len(images) > 0:
                        st.write(f"已上传 {len(images)} 张图片")
                        # 只显示前3张缩略图
                        preview_cols = st.columns(3)
                        for j, image in enumerate(images[:3]):
                            with preview_cols[j % 3]:
                                st.image(image, caption=f"样本 {j + 1}", width=80)

    # 保存类别名称
    st.session_state.class_names_phase1 = class_names

    # 显示数据统计 - 改为水平布局
    if st.session_state.class_images_phase1:
        st.subheader("📊 数据统计")

        # 使用列来水平排列统计信息和警告
        stat_col, warning_col = st.columns([2, 1])

        with stat_col:
            total_images = 0
            warning_messages = []
            for class_name in class_names:
                if class_name in st.session_state.class_images_phase1:
                    count = len(st.session_state.class_images_phase1[class_name])
                    st.write(f"- **{class_name}**: {count} 张图片")
                    total_images += count

                    # 收集警告信息
                    if count < 5:
                        warning_messages.append(f"⚠️ '{class_name}' 只有 {count} 张图片，建议至少上传5张")
                else:
                    warning_messages.append(f"⚠️ '{class_name}' 只有 0 张图片，建议至少上传5张")

            st.write(f"**总计**: {total_images} 张图片")

        with warning_col:
            # 显示所有警告信息
            for warning in warning_messages:
                st.warning(warning)

    # 准备训练数据按钮
    col1, col2 = st.columns(2)
    with col1:
        if st.button("准备训练数据", type="primary", key="phase1_prepare"):
            if not st.session_state.class_images_phase1:
                st.error("请先上传训练数据！")
            else:
                # 检查每类是否至少有5张图片
                valid_data = True
                for class_name in class_names:
                    if class_name not in st.session_state.class_images_phase1 or len(
                            st.session_state.class_images_phase1[class_name]) < 5:
                        st.error(f"'{class_name}' 需要至少5张图片！")
                        valid_data = False

                if valid_data:
                    with st.spinner("正在准备训练数据..."):
                        # 使用进度条显示处理进度
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        # 转换图片为PyTorch张量
                        images = []
                        labels = []

                        total_images = sum(len(st.session_state.class_images_phase1[cn]) for cn in class_names if
                                           cn in st.session_state.class_images_phase1)
                        processed_images = 0

                        for class_idx, class_name in enumerate(class_names):
                            if class_name in st.session_state.class_images_phase1:
                                for image in st.session_state.class_images_phase1[class_name]:
                                    try:
                                        image_tensor = preprocess_image(image)
                                        images.append(image_tensor)
                                        labels.append(class_idx)
                                        processed_images += 1

                                        # 更新进度
                                        progress = processed_images / total_images
                                        progress_bar.progress(progress)
                                        status_text.text(f"处理图片 {processed_images}/{total_images}")

                                    except Exception as e:
                                        st.warning(f"无法处理 {class_name} 的图片: {str(e)}")

                        # 清除进度条
                        progress_bar.empty()
                        status_text.empty()

                        if len(images) < 2:
                            st.error("需要至少2张图片才能进行训练！")
                        else:
                            # 转换为PyTorch张量
                            images_tensor = torch.stack(images)
                            labels_tensor = torch.tensor(labels)

                            # 创建训练数据集
                            train_dataset = AnimalDataset(images_tensor, labels_tensor)
                            st.session_state.train_loader_phase1 = DataLoader(train_dataset, batch_size=8, shuffle=True)

                            st.success(f"✅ 训练数据准备完成！")
                            st.info(f"- 总图片数: {len(images)}")
                            st.info(f"- 图片尺寸: 64x64")
                            st.info(f"- 批处理大小: 8")

    with col2:
        # 清除数据按钮 - 彻底清除所有数据
        if st.button("清除所有数据", key="phase1_clear"):
            # 清除所有图片数据
            st.session_state.class_images_phase1 = {}

            # 清除类别名称
            st.session_state.class_names_phase1 = []

            # 清除训练相关数据
            st.session_state.train_loader_phase1 = None
            st.session_state.model_phase1 = None
            st.session_state.training_history_phase1 = None
            st.session_state.trained_phase1 = False

            # 清除第一阶段的所有缓存
            phase1_keys = [k for k in st.session_state.uploaded_files_cache.keys() if k.startswith('phase1_')]
            for key in phase1_keys:
                del st.session_state.uploaded_files_cache[key]

            # 增加重置计数器，强制创建新的文件上传器
            st.session_state.reset_counter_phase1 += 1

            # 清除所有可能的上传文件相关状态
            for key in list(st.session_state.keys()):
                if "file_uploader" in key or "upload" in key.lower():
                    if not key.startswith("reset_counter"):  # 保留重置计数器
                        del st.session_state[key]

            st.success("所有数据已清除！页面将重新加载...")
            time.sleep(1)
            st.rerun()  # 使用rerun确保完全重置

    # 第二部分：训练模型
    st.markdown(f'<div class="sub-header">🤖 {st.session_state.step2_title}</div>', unsafe_allow_html=True)

    if not st.session_state.class_images_phase1:
        st.warning("请先上传训练数据！")
    else:
        st.markdown("""\

""", unsafe_allow_html=True)

        # 训练参数设置 - 固定为25轮，不显示滑块
        st.write("**训练轮次**: 25轮（固定）")
        epochs = 25

        if st.button("开始训练模型", type="primary", key="phase1_train"):
            if st.session_state.train_loader_phase1 is None:
                st.error("请先准备训练数据！")
            else:
                # 创建模型 - 使用改进的模型
                num_classes = len(st.session_state.class_names_phase1)
                model = ImprovedAnimalCNN(num_classes).to(st.session_state.device)

                # 设置优化器和损失函数 - 使用更小的学习率提高精度
                optimizer = optim.Adam(model.parameters(), lr=0.0005)
                criterion = nn.CrossEntropyLoss()

                # 训练模型
                with st.spinner("模型训练中，请稍候..."):
                    history = train_model(
                        model,
                        st.session_state.train_loader_phase1,
                        criterion,
                        optimizer,
                        epochs,
                        st.session_state.device,
                        phase=1
                    )

                # 保存模型和训练历史
                st.session_state.model_phase1 = model
                st.session_state.training_history_phase1 = history
                st.session_state.trained_phase1 = True

                st.success(f"🎉 物种分类模型训练完成！")

        # 提供模型下载
        if st.session_state.trained_phase1:
            st.subheader("📥 下载模型")
            if st.button("下载模型", key="phase1_download"):
                # 保存模型到字节流
                buffer = io.BytesIO()
                torch.save(st.session_state.model_phase1.state_dict(), buffer)
                buffer.seek(0)

                # 创建下载链接
                b64 = base64.b64encode(buffer.read()).decode()
                href = f'<a href="data:application/octet-stream;base64,{b64}" download="species_classifier.pth">下载PyTorch模型文件</a>'
                st.markdown(href, unsafe_allow_html=True)

    # 第三部分：测试模型
    st.markdown(f'<div class="sub-header">🔍 {st.session_state.step3_title}</div>', unsafe_allow_html=True)

    if not st.session_state.trained_phase1:
        st.warning("请先训练模型！")
    else:
        # 单张图片预测
        st.subheader("单张图片预测")

        # 美化的上传提示
        st.markdown("""\
<div class="upload-hint">
📸 <strong>上传测试图片：</strong> 请点击下方区域选择图片或将图片拖拽到此处
</div>
""", unsafe_allow_html=True)

        # 上传测试图片 - 使用中文提示
        test_image = st.file_uploader(
            "上传测试图片",
            type=['jpg', 'jpeg', 'png'],
            key="phase1_test_uploader",
            help="上传一张未训练过的动物图片进行测试"
        )

        if test_image and st.session_state.model_phase1 is not None:
            image = Image.open(test_image).convert('RGB')

            # 使用两列布局：左侧图片，右侧结果
            test_col1, test_col2 = st.columns([1, 2])

            with test_col1:
                st.image(image, caption="测试图片", width=200)

            with test_col2:
                if st.button("识别动物", type="primary", key="phase1_predict"):
                    # 预处理图片
                    model = st.session_state.model_phase1
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
                        st.warning(f"🔍 识别结果: **{predicted_name}** (置信度中等)")
                    else:
                        st.error(f"🔍 识别结果: **{predicted_name}** (置信度较低)")

                    st.write(f"置信度: {confidence * 100:.2f}%")

                    # 显示所有类别的概率
                    st.subheader("所有类别概率:")
                    for i, class_name in enumerate(st.session_state.class_names_phase1):
                        prob = probabilities[0][i].item() * 100
                        color = "green" if i == predicted_class else "gray"
                        st.write(f"<span style='color:{color};'>{class_name}: {prob:.2f}%</span>",
                                 unsafe_allow_html=True)
                        st.progress(prob / 100)

        # 机器学习基本流程总结 - 放在页面最下方
        st.markdown("---")
        st.markdown('<div class="sub-header">📊 总结：机器学习的基本过程</div>', unsafe_allow_html=True)

        # 添加提交按钮和标题更新功能
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

        with col1:
            step1_input = st.text_input("第一步", value="", key="phase1_step1", placeholder="输入第一步")
        with col2:
            step2_input = st.text_input("第二步", value="", key="phase1_step2", placeholder="输入第二步")
        with col3:
            step3_input = st.text_input("第三步", value="", key="phase1_step3", placeholder="输入第三步")
        with col4:
            st.write("")  # 空行用于对齐
            st.write("")  # 空行用于对齐
            if st.button("提交答案", key="submit_steps"):
                # 检查答案是否正确
                correct_answers = {
                    "step1": ["输入数据", "数据输入"],
                    "step2": ["训练模型", "模型训练"],
                    "step3": ["测试模型", "验证模型", "模型测试", "模型验证"]
                }

                step1_correct = step1_input.strip() in correct_answers["step1"]
                step2_correct = step2_input.strip() in correct_answers["step2"]
                step3_correct = step3_input.strip() in correct_answers["step3"]

                if step1_correct and step2_correct and step3_correct:
                    st.success("答案正确！标题已更新。")
                    # 更新session state中的步骤标题
                    st.session_state.step1_title = "输入数据"
                    st.session_state.step2_title = "训练模型"
                    st.session_state.step3_title = "测试模型"
                else:
                    st.error("答案不正确，请重新输入。")
                    # 提供正确答案提示
                    st.info("**提示：** 正确的三个步骤是：输入数据、训练模型、测试模型（顺序可能不同）")

        # 添加提示
        st.markdown("""\
<div class="upload-hint">
💡 <strong>提示：</strong> 正确的三个步骤是：<strong>输入数据</strong>、<strong>训练模型</strong>、<strong>测试模型</strong>
</div>
""", unsafe_allow_html=True)

# 第二阶段：个体识别 - 改为上下结构
elif page == "第二阶段：个体识别系统":
    st.markdown('<div class="sub-header">🔬 第二阶段：动物个体识别追踪系统</div>', unsafe_allow_html=True)

    st.markdown("""\
<div class="story-box">
<h3>🐼 新挑战：识别特定个体</h3>
<p>现在我们发现保护区内每种动物都有多个个体，特别是大熊猫，我们需要知道"这是哪一只大熊猫？"</p>
<p>巡护员很难仅凭肉眼记住每一只大熊猫的样子，尤其是在图片模糊、光线不好或只拍到局部的情况下。</p>
</div>
""", unsafe_allow_html=True)

    # 第一部分：训练数据
    st.markdown(f'<div class="sub-header">📊 {st.session_state.step1_title}</div>', unsafe_allow_html=True)

    st.markdown("""\
<div class="warning-box">
<strong>注意：</strong> 请为每个个体上传至少5张图片！
</div>
""", unsafe_allow_html=True)

    # 类别设置 - 修改默认值为2
    num_individuals = st.number_input("个体数量", min_value=2, max_value=10, value=2, step=1)

    # 使用两列布局显示个体
    individual_names = []
    for i in range(num_individuals):
        # 生成默认个体名称
        default_name = f"个体{i + 1}"

        # 检查是否已有个体名称，如果有则使用现有的
        if i < len(st.session_state.class_names_phase2):
            default_name = st.session_state.class_names_phase2[i]

        # 每行显示两个个体
        if i % 2 == 0:  # 偶数索引创建新行
            cols = st.columns(2)

        with cols[i % 2]:  # 在当前行的对应列中显示
            with st.container():
                st.markdown(f"**个体 {i + 1}**")
                individual_name = st.text_input(f"个体名称", value=default_name, key=f"individual_name_{i}")
                individual_names.append(individual_name)

                # 确保个体在class_images_phase2中
                if individual_name not in st.session_state.class_images_phase2:
                    st.session_state.class_images_phase2[individual_name] = []

                # 文件上传器 - 使用包含重置计数器的key
                uploader_key = f"individual_uploader_{i}_{st.session_state.reset_counter_phase2}"

                # 美化的上传提示
                st.markdown("""\
<div class="upload-hint">
📸 <strong>上传图片：</strong> 请点击下方区域选择图片或将图片拖拽到此处
</div>
""", unsafe_allow_html=True)

                uploaded_files = st.file_uploader(
                    f"为 '{individual_name}' 上传图片",
                    type=['jpg', 'jpeg', 'png'],
                    accept_multiple_files=True,
                    key=uploader_key,
                    help="支持JPG、JPEG、PNG格式，最多可上传200张图片"
                )

                # 使用改进的上传处理
                if handle_file_upload(uploaded_files, individual_name, i, phase=2):
                    st.success(f"已为 '{individual_name}' 上传 {len(uploaded_files)} 张图片")

                # 显示上传的图片样本（不显示附件列表）
                if st.session_state.class_images_phase2.get(individual_name):
                    images = st.session_state.class_images_phase2[individual_name]
                    if len(images) > 0:
                        st.write(f"已上传 {len(images)} 张图片")
                        # 只显示前3张缩略图
                        preview_cols = st.columns(3)
                        for j, image in enumerate(images[:3]):
                            with preview_cols[j % 3]:
                                st.image(image, caption=f"样本 {j + 1}", width=80)

    # 保存个体名称
    st.session_state.class_names_phase2 = individual_names

    # 显示数据统计 - 改为水平布局
    if st.session_state.class_images_phase2:
        st.subheader("📊 数据统计")

        # 使用列来水平排列统计信息和警告
        stat_col, warning_col = st.columns([2, 1])

        with stat_col:
            total_images = 0
            warning_messages = []
            for individual_name in individual_names:
                if individual_name in st.session_state.class_images_phase2:
                    count = len(st.session_state.class_images_phase2[individual_name])
                    st.write(f"- **{individual_name}**: {count} 张图片")
                    total_images += count

                    # 收集警告信息
                    if count < 5:
                        warning_messages.append(f"⚠️ '{individual_name}' 只有 {count} 张图片，建议至少上传5张")
                else:
                    warning_messages.append(f"⚠️ '{individual_name}' 只有 0 张图片，建议至少上传5张")

            st.write(f"**总计**: {total_images} 张图片")

        with warning_col:
            # 显示所有警告信息
            for warning in warning_messages:
                st.warning(warning)

    # 准备训练数据按钮
    col1, col2 = st.columns(2)
    with col1:
        if st.button("准备训练数据", type="primary", key="phase2_preprocess"):
            if not st.session_state.class_images_phase2:
                st.error("请先上传个体数据！")
            else:
                # 检查每个个体是否至少有5张图片
                valid_data = True
                for individual_name in individual_names:
                    if individual_name not in st.session_state.class_images_phase2 or len(
                            st.session_state.class_images_phase2[individual_name]) < 5:
                        st.error(f"'{individual_name}' 需要至少5张图片！")
                        valid_data = False

                if valid_data:
                    with st.spinner("正在准备训练数据..."):
                        # 使用进度条显示处理进度
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        # 转换图片为PyTorch张量
                        images = []
                        labels = []

                        total_images = sum(
                            len(st.session_state.class_images_phase2[iname]) for iname in individual_names if
                            iname in st.session_state.class_images_phase2)
                        processed_images = 0

                        for individual_idx, individual_name in enumerate(individual_names):
                            if individual_name in st.session_state.class_images_phase2:
                                for image in st.session_state.class_images_phase2[individual_name]:
                                    try:
                                        image_tensor = preprocess_image(image)
                                        images.append(image_tensor)
                                        labels.append(individual_idx)
                                        processed_images += 1

                                        # 更新进度
                                        progress = processed_images / total_images
                                        progress_bar.progress(progress)
                                        status_text.text(f"处理图片 {processed_images}/{total_images}")

                                    except Exception as e:
                                        st.warning(f"无法处理 {individual_name} 的图片: {str(e)}")

                        # 清除进度条
                        progress_bar.empty()
                        status_text.empty()

                        if len(images) < 2:
                            st.error("需要至少2张图片才能进行训练！")
                        else:
                            # 转换为PyTorch张量
                            images_tensor = torch.stack(images)
                            labels_tensor = torch.tensor(labels)

                            # 创建训练数据集
                            train_dataset = AnimalDataset(images_tensor, labels_tensor)
                            st.session_state.train_loader_phase2 = DataLoader(train_dataset, batch_size=8, shuffle=True)

                            st.success(f"✅ 训练数据准备完成！")
                            st.info(f"- 总图片数: {len(images)}")
                            st.info(f"- 图片尺寸: 64x64")
                            st.info(f"- 批处理大小: 8")

    with col2:
        # 清除数据按钮 - 彻底清除所有数据
        if st.button("清除所有数据", key="phase2_clear"):
            # 清除所有图片数据
            st.session_state.class_images_phase2 = {}

            # 清除个体名称
            st.session_state.class_names_phase2 = []

            # 清除训练相关数据
            st.session_state.train_loader_phase2 = None
            st.session_state.model_phase2 = None
            st.session_state.training_history_phase2 = None
            st.session_state.trained_phase2 = False

            # 清除第二阶段的所有缓存
            phase2_keys = [k for k in st.session_state.uploaded_files_cache.keys() if k.startswith('phase2_')]
            for key in phase2_keys:
                del st.session_state.uploaded_files_cache[key]

            # 增加重置计数器，强制创建新的文件上传器
            st.session_state.reset_counter_phase2 += 1

            # 清除所有可能的上传文件相关状态
            for key in list(st.session_state.keys()):
                if "file_uploader" in key or "upload" in key.lower():
                    if not key.startswith("reset_counter"):  # 保留重置计数器
                        del st.session_state[key]

            st.success("所有数据已清除！页面将重新加载...")
            time.sleep(1)
            st.rerun()  # 使用rerun确保完全重置

    # 第二部分：训练模型
    st.markdown(f'<div class="sub-header">🤖 {st.session_state.step2_title}</div>', unsafe_allow_html=True)

    if not st.session_state.class_images_phase2:
        st.warning("请先上传个体数据！")
    else:
        st.markdown("""\

""", unsafe_allow_html=True)

        # 训练参数设置 - 固定为25轮，不显示滑块
        st.write("**训练轮次**: 25轮（固定）")
        epochs = 25

        if st.button("开始训练模型", type="primary", key="phase2_train"):
            if st.session_state.train_loader_phase2 is None:
                st.error("请先准备训练数据！")
            else:
                # 创建模型 - 使用改进的模型
                num_classes = len(st.session_state.class_names_phase2)
                model = ImprovedAnimalCNN(num_classes).to(st.session_state.device)

                # 设置优化器和损失函数
                optimizer = optim.Adam(model.parameters(), lr=0.0005)
                criterion = nn.CrossEntropyLoss()

                # 训练模型
                with st.spinner("个体识别模型训练中..."):
                    history = train_model(
                        model,
                        st.session_state.train_loader_phase2,
                        criterion,
                        optimizer,
                        epochs,
                        st.session_state.device,
                        phase=2
                    )

                # 保存模型和训练历史
                st.session_state.model_phase2 = model
                st.session_state.training_history_phase2 = history
                st.session_state.trained_phase2 = True

                st.success(f"🎉 个体识别模型训练完成！")

        # 提供模型下载
        if st.session_state.trained_phase2:
            st.subheader("📥 下载模型")
            if st.button("下载模型", key="phase2_download"):
                # 保存模型到字节流
                buffer = io.BytesIO()
                torch.save(st.session_state.model_phase2.state_dict(), buffer)
                buffer.seek(0)

                # 创建下载链接
                b64 = base64.b64encode(buffer.read()).decode()
                href = f'<a href="data:application/octet-stream;base64,{b64}" download="individual_recognizer.pth">下载PyTorch模型文件</a>'
                st.markdown(href, unsafe_allow_html=True)

    # 第三部分：测试模型
    st.markdown(f'<div class="sub-header">🔍 {st.session_state.step3_title}</div>', unsafe_allow_html=True)

    if not st.session_state.trained_phase2:
        st.warning("请先训练个体识别模型！")
    else:
        # 单张图片预测
        st.subheader("单张图片预测")

        # 美化的上传提示
        st.markdown("""\
<div class="upload-hint">
📸 <strong>上传测试图片：</strong> 请点击下方区域选择图片或将图片拖拽到此处
</div>
""", unsafe_allow_html=True)

        # 上传测试图片 - 使用中文提示
        test_image = st.file_uploader(
            "上传测试图片",
            type=['jpg', 'jpeg', 'png'],
            key="phase2_test_uploader",
            help="上传一张未训练过的个体图片进行测试"
        )

        if test_image and st.session_state.model_phase2 is not None:
            image = Image.open(test_image).convert('RGB')

            # 使用两列布局：左侧图片，右侧结果
            test_col1, test_col2 = st.columns([1, 2])

            with test_col1:
                st.image(image, caption="测试图片", width=200)

            with test_col2:
                if st.button("识别个体", type="primary", key="phase2_predict"):
                    # 预处理图片
                    model = st.session_state.model_phase2
                    model.eval()

                    # 使用相同的预处理
                    image_tensor = preprocess_image(image).unsqueeze(0)
                    image_tensor = image_tensor.to(st.session_state.device)

                    # 进行预测 - 使用改进的置信度计算
                    with torch.no_grad():
                        output = model(image_tensor)

                    # 改进的置信度计算 - 避免过高置信度
                    # 使用温度缩放来调整置信度分布
                    temperature = 2.0  # 温度参数，>1会平滑概率分布
                    scaled_output = output / temperature
                    probabilities = torch.softmax(scaled_output, dim=1)

                    predicted_class = torch.argmax(probabilities, dim=1).item()
                    confidence = probabilities[0][predicted_class].item()

                    # 如果最大概率和第二大概率很接近，降低置信度
                    sorted_probs, _ = torch.sort(probabilities[0], descending=True)
                    if len(sorted_probs) > 1:
                        gap = sorted_probs[0] - sorted_probs[1]
                        # 如果前两个概率很接近，调整置信度
                        if gap < 0.3:  # 差距小于30%
                            confidence = confidence * 0.7  # 降低置信度

                    # 显示结果
                    predicted_name = st.session_state.class_names_phase2[predicted_class]

                    # 根据置信度显示不同的消息 - 调整阈值
                    if confidence > 0.75:
                        st.success(f"🔍 识别结果: **{predicted_name}**")
                        if confidence > 0.85:
                            st.balloons()
                    elif confidence > 0.5:
                        st.warning(f"🔍 识别结果: **{predicted_name}** (中等置信度)")
                    else:
                        st.error(f"🔍 识别结果: **{predicted_name}** (低置信度，建议检查图片质量或增加训练数据)")

                    st.write(f"置信度: {confidence * 100:.2f}%")

                    # 显示所有个体的概率
                    st.subheader("所有个体概率:")
                    for i, individual_name in enumerate(st.session_state.class_names_phase2):
                        prob = probabilities[0][i].item() * 100
                        color = "green" if i == predicted_class else "gray"
                        st.write(f"<span style='color:{color};'>{individual_name}: {prob:.2f}%</span>",
                                 unsafe_allow_html=True)
                        st.progress(prob / 100)

# 学习单页面 - 根据新的学习单文档完整实现
elif page == "学习单":
    st.markdown('<div class="sub-header">📚 《机器学习之动物保护》学习单</div>', unsafe_allow_html=True)

    # 学习单内容 - 严格按照新文档格式
    st.markdown("""\
""", unsafe_allow_html=True)

    # 一、学习目标
    st.markdown("""\
<div class="learning-question">
<h3>一、学习目标</h3>
<p>1.理解机器学习概念，掌握机器学习的基本过程。</p>
<p>2.用"动物保护AI识别系统"完成物种分类、个体识别模型，总结数据对人工智能的重要性。</p>
<p>3.感知 AI 对动物保护的帮助，能举例说明生活中的机器学习应用。</p>
</div>
""", unsafe_allow_html=True)

    # 二、课堂实践
    st.markdown("""\
<div class="learning-question">
<h3>二、课堂实践</h3>
<p>巡护员在野外需快速区分珍稀动物，减少人工识别时间，避免误判。巡护员如何区分这些动物呢？</p>
<p>动物的特征不同：（列举以下动物的特点）</p>
</div>
""", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        panda_features = st.text_area("大熊猫",
                                      value=st.session_state.learning_answers.get('panda_features', ''),
                                      placeholder="",
                                      height=100,
                                      key="sheet_panda")
        if panda_features:
            st.session_state.learning_answers['panda_features'] = panda_features

    with col2:
        tiger_features = st.text_area("老虎",
                                      value=st.session_state.learning_answers.get('tiger_features', ''),
                                      placeholder="",
                                      height=100,
                                      key="sheet_tiger")
        if tiger_features:
            st.session_state.learning_answers['tiger_features'] = tiger_features

    with col3:
        monkey_features = st.text_area("金丝猴",
                                       value=st.session_state.learning_answers.get('monkey_features', ''),
                                       placeholder="",
                                       height=100,
                                       key="sheet_monkey")
        if monkey_features:
            st.session_state.learning_answers['monkey_features'] = monkey_features

    st.markdown("""\
<div class="learning-question">
<p>机器如何将这些动物和他们的特征对应起来呢？</p>
</div>
""", unsafe_allow_html=True)

    # 机器学习概念
    st.markdown("""\
<div class="learning-question">
<h3>1.机器学习的概念</h3>
<p>机器学习是让机器_________________________，获得知识与技能，从而感知世界、认识世界的技术。</p>
</div>
""", unsafe_allow_html=True)

    ml_concept = st.text_input("填写机器学习概念",
                               value=st.session_state.learning_answers.get('ml_concept', ''),
                               placeholder="",
                               key="sheet_ml_concept")
    if ml_concept:
        st.session_state.learning_answers['ml_concept'] = ml_concept

    # 活动1：探索物种分类模型
    st.markdown("""\
<div class="step-box">
<h3>活动1：探索物种分类模型</h3>
<h4>1.实践操作</h4>
<p>1)打开侧标导航栏，选择第一阶段"物种分类系统"，"第一步"模块</p>
<p>2)在类别名称栏输入动物名称，并上传"学生素材-活动1-上传数据"中对应的动物图片，上传完成后点击"开始训练数据"</p>
<p>3)在"第二步"点击"开始训练模型",等待模型训练完成</p>
<p>4）在"第三步"，上传"学生素材-活动1-测试数据"中的图片，检验识别结果是否正确。</p>
</div>
""", unsafe_allow_html=True)

    # 总结流程
    st.markdown("""\
<div class="learning-question">
<h4>2.总结过程：</h4>
<p>通过活动1的实践探索，填写<strong>机器学习的基本过程：</strong></p>
<p><strong>输入数据</strong></p>
<p><strong>训练模型</strong></p>
<p><strong>验证模型</strong></p>
</div>
""", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        step1 = st.text_input("第一步",
                             value=st.session_state.learning_answers.get('step1', ''),
                             placeholder="输入第一步",
                             key="sheet_step1")
        if step1:
            st.session_state.learning_answers['step1'] = step1

    with col2:
        step2 = st.text_input("第二步",
                             value=st.session_state.learning_answers.get('step2', ''),
                             placeholder="输入第二步",
                             key="sheet_step2")
        if step2:
            st.session_state.learning_answers['step2'] = step2

    with col3:
        step3 = st.text_input("第三步",
                             value=st.session_state.learning_answers.get('step3', ''),
                             placeholder="输入第三步",
                             key="sheet_step3")
        if step3:
            st.session_state.learning_answers['step3'] = step3

    st.markdown("""\
<div class="learning-question">
<p>在"第一阶段物种分类系统"下方的"总结机器学习基本过程"中验证自己的答案。</p>
</div>
""", unsafe_allow_html=True)

    # 活动2：探索个体识别模型
    st.markdown("""\
<div class="step-box">
<h3>活动2 ：探索个体识别模型</h3>
<p>新任务发布："升级AI系统，实现对不同个体的行动轨迹监测，这是精准保护的关键。"</p>
</div>
""", unsafe_allow_html=True)

    # 任务1
    st.markdown("""\
<div class="learning-question">
<h4>任务1：</h4>
<p>打开"动物保护AI识别系统"，打开第二阶段"个体识别系统"，选择"学生素材-活动2-任务1"中的图片，其余步骤和活动1相同。</p>
</div>
""", unsafe_allow_html=True)

    # 测试结果表格
    st.markdown("**测试结果**")

    # 创建任务1的测试结果表格
    task1_data = {
        '测试数据': ['测试数据1（萌兰）', '测试数据2（萌兰）', '测试数据3（萌兰）',
                   '测试数据4（萌兰）', '测试数据5（萌兰）', '测试数据6（花花）',
                   '测试数据7（花花）', '测试数据8（花花）', '测试数据9（花花）', '测试数据10（花花）'],
        '萌兰类别概率': ['', '', '', '', '', '', '', '', '', ''],
        '花花类别概率': ['', '', '', '', '', '', '', '', '', ''],
        '是否正确分类（识别度高与85%）': ['', '', '', '', '', '', '', '', '', '']
    }

    edited_task1_df = st.data_editor(
        pd.DataFrame(task1_data),
        use_container_width=True,
        num_rows="fixed",
        key="task1_table"
    )

    # 准确率计算
    correct_count_task1 = st.number_input("正确识别个数", min_value=0, max_value=10, value=0, key="correct_count_task1")
    total_count_task1 = st.number_input("总测试数据个数", min_value=0, max_value=10, value=10, key="total_count_task1")

    if total_count_task1 > 0:
        accuracy_task1 = (correct_count_task1 / total_count_task1) * 100
        st.write(f"**准确率（正确识别个数/总测试数据个数）=（ {accuracy_task1:.1f} ）%**")

    # 结果和原因
    col1, col2 = st.columns(2)

    with col1:
        individual_result_task1 = st.radio("结果：", ["能识别个体", "不能识别个体"], key="individual_result_task1")

    with col2:
        individual_reason_task1 = st.text_area("原因：",
                                             value=st.session_state.learning_answers.get('individual_reason_task1', ''),
                                             placeholder="分析原因...",
                                             height=100,
                                             key="individual_reason_task1")
        if individual_reason_task1:
            st.session_state.learning_answers['individual_reason_task1'] = individual_reason_task1

    # 任务1总结
    st.markdown("**总结：_______是人工智能的核心要素，它的质量_______，人工智能识别的准确率_______。**")
    col1, col2, col3 = st.columns(3)

    with col1:
        core_element_task1 = st.text_input("核心要素",
                                          value=st.session_state.learning_answers.get('core_element_task1', ''),
                                          placeholder="",
                                          key="core_element_task1")
        if core_element_task1:
            st.session_state.learning_answers['core_element_task1'] = core_element_task1

    with col2:
        quality_effect_task1 = st.text_input("质量影响",
                                            value=st.session_state.learning_answers.get('quality_effect_task1', ''),
                                            placeholder="",
                                            key="quality_effect_task1")
        if quality_effect_task1:
            st.session_state.learning_answers['quality_effect_task1'] = quality_effect_task1

    with col3:
        accuracy_effect_task1 = st.text_input("准确率影响",
                                             value=st.session_state.learning_answers.get('accuracy_effect_task1', ''),
                                             placeholder="",
                                             key="accuracy_effect_task1")
        if accuracy_effect_task1:
            st.session_state.learning_answers['accuracy_effect_task1'] = accuracy_effect_task1

    # 任务2
    st.markdown("""\
<div class="learning-question">
<h4>任务2：分组练习，选择本组的数据进行模型训练。</h4>
</div>
""", unsafe_allow_html=True)

    st.markdown("**测试结果**")

    # 创建任务2的测试结果表格
    task2_data = {
        '测试数据': ['测试数据1（萌兰）', '测试数据2（萌兰）', '测试数据3（萌兰）',
                   '测试数据4（萌兰）', '测试数据5（萌兰）', '测试数据6（花花）',
                   '测试数据7（花花）', '测试数据8（花花）', '测试数据9（花花）', '测试数据10（花花）'],
        '萌兰类别概率': ['', '', '', '', '', '', '', '', '', ''],
        '花花类别概率': ['', '', '', '', '', '', '', '', '', ''],
        '是否正确分类（识别度高与85%）': ['', '', '', '', '', '', '', '', '', '']
    }

    edited_task2_df = st.data_editor(
        pd.DataFrame(task2_data),
        use_container_width=True,
        num_rows="fixed",
        key="task2_table"
    )

    # 准确率计算
    correct_count_task2 = st.number_input("正确识别个数", min_value=0, max_value=10, value=0, key="correct_count_task2")
    total_count_task2 = st.number_input("总测试数据个数", min_value=0, max_value=10, value=10, key="total_count_task2")

    if total_count_task2 > 0:
        accuracy_task2 = (correct_count_task2 / total_count_task2) * 100
        st.write(f"**准确率（正确识别个数/总测试数据个数）=（ {accuracy_task2:.1f} ）%**")

    # 结果和原因
    col1, col2 = st.columns(2)

    with col1:
        individual_result_task2 = st.radio("结果：", ["能识别个体", "不能识别个体"], key="individual_result_task2")

    with col2:
        individual_reason_task2 = st.text_area("原因：",
                                             value=st.session_state.learning_answers.get('individual_reason_task2', ''),
                                             placeholder="分析原因...",
                                             height=100,
                                             key="individual_reason_task2")
        if individual_reason_task2:
            st.session_state.learning_answers['individual_reason_task2'] = individual_reason_task2

    # 任务2总结
    st.markdown("**总结：_______是人工智能的核心要素，它的数量_______，人工智能识别的准确率_______。**")
    col1, col2, col3 = st.columns(3)

    with col1:
        core_element_task2 = st.text_input("核心要素",
                                          value=st.session_state.learning_answers.get('core_element_task2', ''),
                                          placeholder="",
                                          key="core_element_task2")
        if core_element_task2:
            st.session_state.learning_answers['core_element_task2'] = core_element_task2

    with col2:
        quantity_effect_task2 = st.text_input("数量影响",
                                             value=st.session_state.learning_answers.get('quantity_effect_task2', ''),
                                             placeholder="",
                                             key="quantity_effect_task2")
        if quantity_effect_task2:
            st.session_state.learning_answers['quantity_effect_task2'] = quantity_effect_task2

    with col3:
        accuracy_effect_task2 = st.text_input("准确率影响",
                                             value=st.session_state.learning_answers.get('accuracy_effect_task2', ''),
                                             placeholder="",
                                             key="accuracy_effect_task2")
        if accuracy_effect_task2:
            st.session_state.learning_answers['accuracy_effect_task2'] = accuracy_effect_task2

    # 任务3
    st.markdown("""\
<div class="learning-question">
<h4>任务3：分组训练，选择本组的数据进行模型训练。</h4>
</div>
""", unsafe_allow_html=True)

    # 创建任务3的测试结果表格
    task3_data = {
        '测试数据': ['测试数据1（萌兰）', '测试数据2（萌兰）', '测试数据3（萌兰）',
                   '测试数据4（萌兰）', '测试数据5（萌兰）', '测试数据6（花花）',
                   '测试数据7（花花）', '测试数据8（花花）', '测试数据9（花花）', '测试数据10（花花）'],
        '萌兰类别概率': ['', '', '', '', '', '', '', '', '', ''],
        '花花类别概率': ['', '', '', '', '', '', '', '', '', ''],
        '是否正确分类（识别度高与85%）': ['', '', '', '', '', '', '', '', '', '']
    }

    edited_task3_df = st.data_editor(
        pd.DataFrame(task3_data),
        use_container_width=True,
        num_rows="fixed",
        key="task3_table"
    )

    # 准确率计算
    correct_count_task3 = st.number_input("正确识别个数", min_value=0, max_value=10, value=0, key="correct_count_task3")
    total_count_task3 = st.number_input("总测试数据个数", min_value=0, max_value=10, value=10, key="total_count_task3")

    if total_count_task3 > 0:
        accuracy_task3 = (correct_count_task3 / total_count_task3) * 100
        st.write(f"**准确率（正确识别个数/总测试数据个数）=（ {accuracy_task3:.1f} ）%**")

    st.markdown("**总结：**")
    summary_task3 = st.text_area("填写总结",
                               value=st.session_state.learning_answers.get('summary_task3', ''),
                               placeholder="填写任务3的总结...",
                               height=100,
                               key="task3_summary")
    if summary_task3:
        st.session_state.learning_answers['summary_task3'] = summary_task3

    st.success("**以上任务圆满完成，恭喜你成为优秀的野生动物保护AI研究员！**")

    # 三、课堂总结
    st.markdown("""\
<div class="learning-question">
<h3>三、课堂总结（梳理与反思）</h3>
</div>
""", unsafe_allow_html=True)

    st.markdown("**1.这节课你学到的核心知识点：**")
    key_points = st.text_area("核心知识点",
                              value=st.session_state.learning_answers.get('key_points', ''),
                              placeholder="写下你学到的核心知识点...",
                              height=100,
                              key="key_points")
    if key_points:
        st.session_state.learning_answers['key_points'] = key_points

    st.markdown("**2.关于机器学习，你还有哪些疑问？**")
    questions = st.text_area("疑问",
                             value=st.session_state.learning_answers.get('questions', ''),
                             placeholder="写下你的疑问...",
                             height=100,
                             key="questions")
    if questions:
        st.session_state.learning_answers['questions'] = questions

    # 四、拓展学习
    st.markdown("""\
<div class="learning-question">
<h3>四、拓展学习</h3>
<p>思考：机器学习在生活中的应用有哪些？</p>
</div>
""", unsafe_allow_html=True)

    ml_applications = st.text_area("机器学习应用",
                                   value=st.session_state.learning_answers.get('ml_applications', ''),
                                   placeholder="列举机器学习在生活中的应用...",
                                   height=100,
                                   key="ml_applications")
    if ml_applications:
        st.session_state.learning_answers['ml_applications'] = ml_applications

    # 五、学习评价
    st.markdown("""\
<div class="learning-question">
<h3>五、学习评价</h3>
</div>
""", unsafe_allow_html=True)

    st.markdown("**1.经过本课的学习，你有哪些收获呢？我们快速扫描一遍，对所学内容进行整理。**")
    harvest = st.text_area("学习收获",
                           value=st.session_state.learning_answers.get('harvest', ''),
                           placeholder="写下你的收获...",
                           height=100,
                           key="harvest")
    if harvest:
        st.session_state.learning_answers['harvest'] = harvest

    st.markdown("**2.活动评价。(学生自我评价，根据评价结果将相应数量的五角星，五颗星为最佳成绩。)**")

    # 学习评价表格
    evaluation_data = {
        '描述': [
            '(1)我觉得认识了机器学习的概念及基本过程，了解数据在人工智能领域的重要性。',
            '(2)在课堂互动环节中，我有积极地参与到课堂的互动中来。'
        ],
        '学生自评': ['☆☆☆☆☆', '☆☆☆☆☆']
    }

    st.table(pd.DataFrame(evaluation_data))

    col1, col2 = st.columns(2)

    with col1:
        rating1 = st.slider("(1)评分", 1, 5,
                           value=st.session_state.learning_answers.get('rating1', 3),
                           key="rating1")
        st.write("⭐" * rating1)
        st.session_state.learning_answers['rating1'] = rating1

    with col2:
        rating2 = st.slider("(2)评分", 1, 5,
                           value=st.session_state.learning_answers.get('rating2', 3),
                           key="rating2")
        st.write("⭐" * rating2)
        st.session_state.learning_answers['rating2'] = rating2

    # 保存学习单答案
    if st.button("保存学习单答案", type="primary"):
        # 更新所有答案到session state
        st.session_state.learning_answers.update({
            'panda_features': panda_features,
            'tiger_features': tiger_features,
            'monkey_features': monkey_features,
            'ml_concept': ml_concept,
            'step1': step1,
            'step2': step2,
            'step3': step3,
            'individual_reason_task1': individual_reason_task1,
            'core_element_task1': core_element_task1,
            'quality_effect_task1': quality_effect_task1,
            'accuracy_effect_task1': accuracy_effect_task1,
            'individual_reason_task2': individual_reason_task2,
            'core_element_task2': core_element_task2,
            'quantity_effect_task2': quantity_effect_task2,
            'accuracy_effect_task2': accuracy_effect_task2,
            'summary_task3': summary_task3,
            'key_points': key_points,
            'questions': questions,
            'ml_applications': ml_applications,
            'harvest': harvest,
            'rating1': rating1,
            'rating2': rating2
        })
        st.success("学习单答案已保存！")

    st.markdown("</div>", unsafe_allow_html=True)  # 结束learning-sheet div

# AI研究员证书页面
elif page == "AI研究员证书":
    st.markdown('<div class="sub-header">🏆 AI研究员成就证书</div>', unsafe_allow_html=True)

    # 创建证书
    st.markdown("""\
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
<h3 style="color: #F57C00;">机器学习的基本过程</h3>
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
        unsafe_allow_html=True)
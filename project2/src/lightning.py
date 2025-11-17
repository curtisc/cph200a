import lightning.pytorch as pl
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import torchvision
from src.cindex import concordance_index
from src.profiling import get_global_profiler

class Classifer(pl.LightningModule):
    """
    Base classifier class using PyTorch Lightning framework.
    
    This class provides the common training/validation/test loop structure
    and metrics computation for classification tasks. Subclasses need to
    implement the forward() method to define their specific architecture.
    """
    
    def __init__(self, num_classes=9, init_lr=1e-4, positive_class: int | None = None, class_weights: torch.Tensor | None = None):
        """
        Initialize the base classifier.
        
        Args:
            num_classes (int): Number of classes for classification
            init_lr (float): Initial learning rate for optimizer
        """
        super().__init__()
        self.init_lr = init_lr
        self.num_classes = num_classes
        # For binary problems, define which class index is considered positive for AUC
        # Default to last class (num_classes-1) if not specified
        self.positive_class = (num_classes - 1) if positive_class is None else positive_class

        # Cross-entropy loss (optionally weighted for class imbalance)
        self.loss = nn.CrossEntropyLoss(weight=class_weights)

        # Metrics with proper distributed training configuration
        # sync_on_compute=True: synchronize across GPUs before computing final metric
        # dist_sync_on_step=False: don't sync on every step (only at epoch end)
        self.accuracy = torchmetrics.Accuracy(
            task="multiclass",
            num_classes=self.num_classes,
            sync_on_compute=True,
            dist_sync_on_step=False
        )
        
        # AUC (Area Under ROC Curve) - handles both binary and multiclass
        self.auc = torchmetrics.AUROC(
            task="binary" if self.num_classes == 2 else "multiclass",
            num_classes=self.num_classes,
            sync_on_compute=True,
            dist_sync_on_step=False
        )

        # Store outputs from each batch to compute epoch-level metrics
        # This allows us to compute metrics over the entire epoch, not just per-batch
        self.training_outputs = []
        self.validation_outputs = []
        self.test_outputs = []

    def _log_epoch_auc(self, stage: str, probs: torch.Tensor, targets: torch.Tensor) -> None:
        """
        Log AUROC in a numerically safe way, skipping computation when class labels
        are missing (e.g. no positive samples present) to avoid torchmetrics warnings.
        """
        targets = targets.view(-1)
        # Nothing to log if there are no targets accumulated
        if targets.numel() == 0:
            self.auc.reset()
            return

        if self.num_classes == 2:
            unique_targets = torch.unique(targets)
            if unique_targets.numel() < 2:
                # Skip AUROC computation when we only observe a single class
                self.log(f"{stage}_auc_nan", 1.0, sync_dist=True, prog_bar=False)
                self.auc.reset()
                return

        auc_value = self.auc(probs, targets)
        self.log(f"{stage}_auc", auc_value, sync_dist=True, prog_bar=True)
        self.auc.reset()

    def get_xy(self, batch):
        """
        Extract input features (x) and labels (y) from batch data.
        
        Handles two different data formats:
        1. Tuple/list format: (images, labels) - used by PathMNIST
        2. Dictionary format: {"x": images, "y_seq": labels} - used by NLST
        
        Args:
            batch: Batch data from DataLoader
            
        Returns:
            tuple: (x, y) where x is input tensor and y is label tensor
        """
        if isinstance(batch, (list, tuple)):
            # Standard format: (images, labels)
            x, y = batch[0], batch[1]
        else:
            # Dictionary format from NLST dataset
            assert isinstance(batch, dict)
            x, y = batch["x"], batch["y_seq"][:, 0]

        # Ensure labels are long integers and flattened for CrossEntropyLoss
        return x, y.to(torch.long).view(-1)

    def training_step(self, batch, batch_idx):
        """
        Single training step - called for each batch during training.

        Args:
            batch: Batch of training data
            batch_idx: Index of current batch

        Returns:
            torch.Tensor: Loss value for this batch
        """
        # Get or create profiler (needed for DDP where each rank is separate process)
        profiler = get_global_profiler()

        # Debug output on first batch
        if batch_idx == 0:
            print(f"[DEBUG batch_idx=0] profiler from get_global_profiler: {profiler}")
            print(f"[DEBUG batch_idx=0] hasattr enable_profiling: {hasattr(self, 'enable_profiling')}")
            print(f"[DEBUG batch_idx=0] enable_profiling value: {getattr(self, 'enable_profiling', 'NOT_SET')}")

        if profiler is None and hasattr(self, 'enable_profiling') and self.enable_profiling:
            # Create profiler in this DDP rank
            from src.profiling import BottleneckProfiler, set_global_profiler
            profiler = BottleneckProfiler(enabled=True, log_interval=getattr(self, 'profile_log_interval', 10))
            set_global_profiler(profiler)
            print(f"[INFO] Profiler created in training_step (batch_idx={batch_idx})")

        # Extract inputs and labels from batch
        x, y = self.get_xy(batch)

        # Profile CPU->GPU transfer
        if profiler and not x.is_cuda:
            with profiler.profile_section('cpu_to_gpu'):
                x = x.cuda()
                y = y.cuda()
        else:
            x = x.cuda() if not x.is_cuda else x
            y = y.cuda() if not y.is_cuda else y

        # Forward pass: get model predictions
        if profiler:
            with profiler.profile_section('gpu_forward'):
                y_hat = self.forward(x)
        else:
            y_hat = self.forward(x)

        # Compute loss using cross-entropy
        loss = self.loss(y_hat, y)

        # Log metrics to progress bar and logger
        # on_step=True: log every step, on_epoch=False: don't aggregate over epoch
        # No sync_dist on training steps to avoid distributed training hangs
        if profiler:
            with profiler.profile_section('metric_computation'):
                self.log('train_acc', self.accuracy(y_hat, y), prog_bar=True, on_step=True, on_epoch=False)
                self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=False)
        else:
            self.log('train_acc', self.accuracy(y_hat, y), prog_bar=True, on_step=True, on_epoch=False)
            self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=False)

        # Store predictions and labels for epoch-end AUC computation
        self.training_outputs.append({
            "y_hat": y_hat,
            "y": y
        })

        # Call profiler step
        if profiler:
            profiler.step()
            # Log profiling metrics to wandb
            if batch_idx % profiler.log_interval == 0:
                self.log_dict(profiler.get_summary_dict(), on_step=True, on_epoch=False)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Single validation step - called for each batch during validation.
        
        Args:
            batch: Batch of validation data
            batch_idx: Index of current batch
            
        Returns:
            torch.Tensor: Loss value for this batch
        """
        x, y = self.get_xy(batch)

        # Forward pass (no gradient computation in validation)
        y_hat = self.forward(x)  # (B, num_classes)

        loss = self.loss(y_hat, y)

        # Log validation metrics
        # on_step=False, on_epoch=True: only log aggregated values at epoch end
        # sync_dist=True: synchronize across GPUs for accurate distributed metrics
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_acc", self.accuracy(y_hat, y), prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

        # Store for epoch-end AUC computation
        self.validation_outputs.append({
            "y_hat": y_hat,
            "y": y
        })
        return loss

    def test_step(self, batch, batch_idx):
        """
        Single test step - called for each batch during testing.
        
        Args:
            batch: Batch of test data
            batch_idx: Index of current batch
            
        Returns:
            torch.Tensor: Loss value for this batch
        """
        x, y = self.get_xy(batch)

        # Forward pass for testing
        y_hat = self.forward(x)

        loss = self.loss(y_hat, y)

        # Log test metrics (aggregated at epoch end)
        self.log('test_loss', loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log('test_acc', self.accuracy(y_hat, y), prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

        # Store for epoch-end AUC computation
        self.test_outputs.append({
            "y_hat": y_hat,
            "y": y
        })
        return loss
    def on_train_epoch_end(self):
        """
        Called at the end of each training epoch.
        Computes AUC metric over all training batches from the epoch.
        """
        # Concatenate all predictions and labels from the epoch
        y_hat = torch.cat([o["y_hat"] for o in self.training_outputs])
        y = torch.cat([o["y"] for o in self.training_outputs])

        # Convert logits to probabilities for AUC computation
        if self.num_classes == 2:
            # Binary classification: use probability of configured positive class
            probs = F.softmax(y_hat, dim=-1)[:, self.positive_class]
        else:
            # Multi-class: use full probability distribution
            probs = F.softmax(y_hat, dim=-1)

        # Log AUC metric (synchronized across GPUs)
        self._log_epoch_auc("train", probs, y)

        # Clear stored outputs to free memory
        self.training_outputs = []

        # Print profiling summary at epoch end
        profiler = get_global_profiler()
        if profiler:
            profiler.epoch_end()

    def on_validation_epoch_end(self):
        """
        Called at the end of each validation epoch.
        Computes AUC metric over all validation batches from the epoch.
        """
        # Concatenate all predictions and labels from the epoch
        y_hat = torch.cat([o["y_hat"] for o in self.validation_outputs])
        y = torch.cat([o["y"] for o in self.validation_outputs])
        
        # Convert logits to probabilities
        if self.num_classes == 2:
            probs = F.softmax(y_hat, dim=-1)[:, self.positive_class]
        else:
            probs = F.softmax(y_hat, dim=-1)
            
        # Log validation AUC
        self._log_epoch_auc("val", probs, y.view(-1))
        
        # Clear stored outputs
        self.validation_outputs = []

    def on_test_epoch_end(self):
        """
        Called at the end of testing.
        Computes AUC metric over all test batches.
        """
        # Handle case where no test outputs were stored (empty test set)
        if len(self.test_outputs) == 0:
            return
            
        # Concatenate all test predictions and labels
        y_hat = torch.cat([o["y_hat"] for o in self.test_outputs])
        y = torch.cat([o["y"] for o in self.test_outputs])

        # Convert logits to probabilities
        if self.num_classes == 2:
            probs = F.softmax(y_hat, dim=-1)[:, self.positive_class]
        else:
            probs = F.softmax(y_hat, dim=-1)

        # Log test AUC
        self._log_epoch_auc("test", probs, y.view(-1))
        
        # Clear stored outputs
        self.test_outputs = []

    def configure_optimizers(self):
        """
        Configure optimizer and learning rate scheduler for training.

        Uses AdamW with weight decay and cosine annealing with warmup for better convergence.

        Returns:
            dict: Dictionary containing optimizer and scheduler configuration
        """
        # Filter to only trainable parameters (important when backbone is frozen)
        trainable_params = [p for p in self.parameters() if p.requires_grad]

        # AdamW optimizer: Adam with decoupled weight decay regularization
        # Better generalization than standard Adam
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.init_lr,
            weight_decay=1e-4,  # L2 regularization
            betas=(0.9, 0.999)
        )

        # OPTION 1: ReduceLROnPlateau (adaptive, metric-based)
        # Good when you're not sure about training duration
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',        # Monitor for decreasing validation loss (or 'max' for AUC)
            factor=0.5,        # Multiply LR by 0.5 when plateau detected
            patience=5,        # Wait 5 epochs before reducing LR
            min_lr=1e-7        # Don't reduce below this
        )

        # OPTION 2: CosineAnnealingLR with warmup (better for long training)
        # Uncomment to use this instead:
        # from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
        # warmup = LinearLR(optimizer, start_factor=0.1, total_iters=5)
        # cosine = CosineAnnealingLR(optimizer, T_max=95, eta_min=1e-7)
        # scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[5])

        # Return Lightning-compatible configuration
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",  # Change to "val_auc" if monitoring AUC
                "interval": "epoch",
                "frequency": 1
            }
        }


class Linear(Classifer):
    """
    Linear classifier (also called Logistic Regression for classification).
    
    This is the simplest possible neural network - just a single linear layer
    that directly maps input features to class predictions. It learns a linear
    decision boundary in the input space. Despite its simplicity, it serves as
    an important baseline and can work well when features are already good
    representations or when data is linearly separable.
    
    Mathematical form: y = Wx + b
    where W is the weight matrix and b is the bias vector.
    """
    
    def __init__(self, input_dim=28*28*3, num_classes=9, init_lr=1e-3, **kwargs):
        """
        Initialize the linear classifier.
        
        Args:
            input_dim (int): Size of flattened input features (e.g., 28*28*3 for RGB images)
            num_classes (int): Number of output classes for classification
            init_lr (float): Initial learning rate
        """
        super().__init__(num_classes=num_classes, init_lr=init_lr)
        self.save_hyperparameters()
        
        # Single linear transformation: input_dim -> num_classes
        # This creates a weight matrix of size (input_dim, num_classes) and bias vector
        # Each output neuron represents one class
        self.model = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        """
        Forward pass through the linear classifier.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes)
        """
        # Flatten the input: (B, C, H, W) -> (B, C*H*W)
        # Linear layers expect 2D input: (batch_size, features)
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)
        
        # Apply linear transformation: y = Wx + b
        # This computes a weighted sum of all input pixels for each class
        logits = self.model(x_flat)
        
        return logits


class MLP(Classifer):
    """
    Multi-Layer Perceptron (MLP) classifier for image classification.
    
    This is a fully-connected neural network that flattens input images
    and passes them through a series of linear layers with ReLU activations.
    """
    
    def __init__(self, input_dim=28*28*3, hidden_dim=128, num_layers=1, num_classes=9, use_bn=False, init_lr = 1e-3, **kwargs):
        """
        Initialize the MLP model.
        
        Args:
            input_dim (int): Size of flattened input (e.g., 28*28*3 for RGB images)
            hidden_dim (int): Number of neurons in each hidden layer
            num_layers (int): Number of hidden layers
            num_classes (int): Number of output classes for classification
            use_bn (bool): Whether to use batch normalization
            init_lr (float): Initial learning rate
        """
        super().__init__(num_classes=num_classes, init_lr=init_lr)
        self.save_hyperparameters()

        self.hidden_dim = hidden_dim
        self.use_bn = use_bn

        # Build the MLP architecture layer by layer
        layers = []        
        in_dim = input_dim
        
        # Add hidden layers
        for i in range(num_layers):
            # Linear transformation: y = Wx + b
            layers.append(nn.Linear(in_dim, hidden_dim))
            
            # Optional batch normalization for training stability
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            # ReLU activation: f(x) = max(0, x)
            layers.append(nn.ReLU())
            
            # Update input dimension for next layer
            in_dim = hidden_dim
            
        # Final output layer (no activation - raw logits for classification)
        layers.append(nn.Linear(hidden_dim, num_classes))
        
        # Combine all layers into a sequential model
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass through the MLP.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes)
        """
        # Get batch size from input tensor
        batch_size, channels, width, height = x.size()
        
        # Flatten the image: (B, C, H, W) -> (B, C*H*W)
        # This converts 2D/3D image data into a 1D vector for each sample
        x = x.view(batch_size, -1)
        
        # Pass through the MLP layers
        return self.model(x)



class CNN(Classifer):
    """
    Convolutional Neural Network (CNN) classifier for image classification.
    
    CNNs use convolutional layers to detect spatial patterns and features
    in images, making them much more effective than MLPs for image tasks.
    The architecture consists of convolutional layers followed by pooling
    and a final fully-connected classifier head.
    """
    
    def __init__(self, input_channels=3, hidden_dim=128, num_layers=2, num_classes=9, use_bn=True, init_lr=1e-3, **kwargs):
        """
        Initialize the CNN model.
        
        Args:
            input_channels (int): Number of input channels (1 for grayscale, 3 for RGB)
            hidden_dim (int): Number of channels in convolutional layers
            num_layers (int): Number of convolutional blocks
            num_classes (int): Number of output classes for classification
            use_bn (bool): Whether to use batch normalization
            init_lr (float): Initial learning rate
        """
        super().__init__(num_classes=num_classes, init_lr=init_lr)
        self.save_hyperparameters()

        self.hidden_dim = hidden_dim
        self.use_bn = use_bn
        self.num_layers = num_layers

        # Build convolutional feature extractor
        conv_layers = []
        in_channels = input_channels
        
        for i in range(num_layers):
            # Convolutional layer: applies learnable filters to detect features
            conv_layers.append(nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1))
            
            # Optional batch normalization for training stability
            if use_bn:
                conv_layers.append(nn.BatchNorm2d(hidden_dim))
            
            # ReLU activation for non-linearity
            conv_layers.append(nn.ReLU(inplace=True))
            
            # Max pooling: reduces spatial dimensions while keeping important features
            conv_layers.append(nn.MaxPool2d(2, 2))
            
            # Update input channels for next layer
            in_channels = hidden_dim
            
            # Increase number of channels in deeper layers (common CNN pattern)
            if i < num_layers - 1:
                hidden_dim = min(hidden_dim * 2, 512)  # Cap at 512 channels
        
        self.conv_layers = nn.Sequential(*conv_layers)
        
        # Global average pooling: reduces feature maps to single values
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classifier head: fully-connected layer for final classification
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),  # Dropout for regularization
            nn.Linear(in_channels, num_classes)
        )

    def forward(self, x):
        """
        Forward pass through the CNN.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes)
        """
        # Extract features using convolutional layers
        # Each conv block: Conv2d -> BatchNorm -> ReLU -> MaxPool
        features = self.conv_layers(x)  # (B, C, H', W') where H', W' are reduced
        
        # Global average pooling: (B, C, H', W') -> (B, C, 1, 1)
        # This summarizes each feature map into a single value
        pooled = self.global_pool(features)
        
        # Flatten for fully-connected layer: (B, C, 1, 1) -> (B, C)
        flattened = pooled.view(pooled.size(0), -1)
        
        # Final classification: (B, C) -> (B, num_classes)
        logits = self.classifier(flattened)
        
        return logits

class ResNet18(Classifer):
    """
    ResNet-18 classifier for image classification.
    
    ResNet-18 is a 18-layer deep residual network that solves the vanishing 
    gradient problem using skip connections (residual connections). The key 
    innovation is residual blocks that learn residual mappings F(x) instead 
    of direct mappings H(x), where H(x) = F(x) + x.
    
    This implementation offers two modes:
    1. Pretrained: Uses ImageNet weights for transfer learning (recommended)
    2. From scratch: Trains the entire network from random initialization
    """
    
    def __init__(self, pretrained=True, num_classes=9, init_lr=1e-3, **kwargs):
        """
        Initialize ResNet-18 model.
        
        Args:
            pretrained (bool): Whether to use ImageNet pre-trained weights
                - True: Transfer learning - faster convergence, better performance
                - False: Train from scratch - slower, may need more data
            num_classes (int): Number of output classes for classification
            init_lr (float): Initial learning rate
        """
        super().__init__(num_classes=num_classes, init_lr=init_lr)
        self.save_hyperparameters()

        self.pretrained = pretrained
        
        # Load ResNet-18 from torchvision
        # Note: Use weights parameter for modern torchvision, but support deprecated pretrained for compatibility
        try:
            # Try modern API first (torchvision >= 0.13)
            if pretrained:
                weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
                self.backbone = torchvision.models.resnet18(weights=weights)
                print("Using ImageNet pre-trained ResNet-18 (modern API)")
            else:
                self.backbone = torchvision.models.resnet18(weights=None)
                print("Training ResNet-18 from scratch")
        except AttributeError:
            # Fallback to deprecated API for older torchvision versions
            if pretrained:
                self.backbone = torchvision.models.resnet18(pretrained=True)
                print("Using ImageNet pre-trained ResNet-18 (deprecated API)")
            else:
                self.backbone = torchvision.models.resnet18(pretrained=False)
                print("Training ResNet-18 from scratch")
        
        # ResNet-18 final layer has 512 features
        num_features = self.backbone.fc.in_features  # 512 for ResNet-18
        
        # Replace the final classifier layer for our specific task
        # Original: 512 -> 1000 (ImageNet classes)
        # New: 512 -> num_classes (our task)
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),  # Regularization to prevent overfitting
            nn.Linear(num_features, num_classes)
        )
        
        # Setup transfer learning strategy if using pretrained weights
        if pretrained:
            self._setup_transfer_learning()
    
    def _setup_transfer_learning(self):
        """
        Setup transfer learning by initially freezing the backbone.
        
        Transfer learning strategy:
        1. Freeze all pre-trained layers (backbone)
        2. Only train the new classifier layer
        3. Later can unfreeze backbone for fine-tuning
        
        This prevents destroying good ImageNet features during initial training.
        """
        # Freeze all backbone parameters (conv layers, batch norms, etc.)
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Unfreeze only the final classifier layer we just created
        for param in self.backbone.fc.parameters():
            param.requires_grad = True
            
        print("Transfer learning setup: backbone frozen, classifier trainable")
    
    def unfreeze_backbone(self):
        """
        Unfreeze the entire backbone for fine-tuning.
        
        Call this method after initial classifier training to fine-tune
        the entire network with a lower learning rate.
        """
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("Backbone unfrozen - ready for fine-tuning")

    def forward(self, x):
        """
        Forward pass through ResNet-18.
        
        ResNet-18 Architecture:
        1. Initial: 7x7 conv -> BatchNorm -> ReLU -> 3x3 MaxPool
        2. Layer1: 2 residual blocks (64 channels)
        3. Layer2: 2 residual blocks (128 channels) 
        4. Layer3: 2 residual blocks (256 channels)
        5. Layer4: 2 residual blocks (512 channels)
        6. Global Average Pool -> FC layer
        
        Each residual block: Conv -> BatchNorm -> ReLU -> Conv -> BatchNorm -> (+skip) -> ReLU
        
        Args:
            x (torch.Tensor): Input images of shape (batch_size, 3, H, W)
            
        Returns:
            torch.Tensor: Class logits of shape (batch_size, num_classes)
        """
        # CRITICAL: Apply ImageNet normalization for pretrained models
        # ResNet-18 pretrained on ImageNet expects this specific normalization
        if self.pretrained:
            # ImageNet normalization stats: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            mean = torch.tensor([0.485, 0.456, 0.406], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
            # Input x is already in [0, 1] range from ToTensor(), so normalize
            x = (x - mean) / std
        
        # Resize to 224x224 if needed (ResNet-18 expects 224x224 for optimal performance)
        # Use interpolation to handle different input sizes gracefully
        if x.shape[2] != 224 or x.shape[3] != 224:
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        
        # The torchvision backbone handles the entire ResNet-18 forward pass
        # including all residual blocks, skip connections, and final classification
        return self.backbone(x)


class SEBlock3D(nn.Module):
    """
    Squeeze-and-Excitation block for 3D convolutions.
    Adaptively recalibrates channel-wise feature responses.
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool3d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _, _ = x.size()
        # Squeeze: global spatial pooling
        y = self.squeeze(x).view(b, c)
        # Excitation: channel-wise attention weights
        y = self.excitation(y).view(b, c, 1, 1, 1)
        # Scale the input features
        return x * y.expand_as(x)


class SpatialAttentionPooling3D(nn.Module):
    """
    Learnable spatial attention pooling for 3D features.
    Generates a soft attention map over spatial dimensions (D, H, W) to weight features
    before global pooling, allowing the model to focus on relevant regions.
    """
    def __init__(self, in_channels: int):
        super().__init__()
        # 1x1x1 convolution to generate spatial attention scores
        self.attention_conv = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // 8, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, return_attention: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Feature tensor of shape (B, C, D, H, W)
            return_attention: Whether to return the attention map

        Returns:
            pooled: Attention-weighted pooled features (B, C)
            attention_map (optional): Spatial attention weights (B, 1, D, H, W)
        """
        # Generate spatial attention map
        attention_scores = self.attention_conv(x)  # (B, 1, D, H, W)
        attention_weights = torch.sigmoid(attention_scores)  # Normalize to [0, 1]

        # Apply attention weighting and pool
        weighted_features = x * attention_weights  # Broadcast multiply
        pooled = torch.sum(weighted_features, dim=(2, 3, 4))  # (B, C)

        # Normalize by total attention mass to maintain scale
        attention_mass = torch.sum(attention_weights, dim=(2, 3, 4)) + 1e-6  # (B, 1)
        pooled = pooled / attention_mass  # (B, C) / (B, 1) -> (B, C)

        if return_attention:
            return pooled, attention_weights
        return pooled


class BasicBlock3D(nn.Module):
    """
    3D variant of the standard ResNet BasicBlock.
    Maintains residual connections while operating over volumetric data.
    Now with optional Squeeze-and-Excitation attention.
    """
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: tuple[int, int, int] | int = (1, 1, 1),
                 downsample: nn.Module | None = None, use_se: bool = True):
        super().__init__()
        stride = self._to_3tuple(stride)

        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample

        # Add SE block for channel attention
        self.se = SEBlock3D(planes) if use_se else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Apply SE attention
        if self.se is not None:
            out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return self.relu(out)

    @staticmethod
    def _to_3tuple(value: tuple[int, int, int] | int) -> tuple[int, int, int]:
        if isinstance(value, tuple):
            return value
        return (value, value, value)


class ResNet18_3D(Classifer):
    """
    ResNet-18 inflated to 3D for volumetric CT data.

    Supports optional weight inflation from 2D ImageNet pretraining to accelerate convergence.
    """

    def __init__(
        self,
        input_channels: int = 1,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        num_classes: int = 2,
        init_lr: float = 5e-4,
        use_se: bool = True,  # Enable Squeeze-and-Excitation attention
        use_attention_pooling: bool = False,  # Use spatial attention pooling instead of avg pooling
        use_localization_reg: bool = False,  # Use localization masks as regularization
        localization_reg_weight: float = 0.1,  # Weight for localization regularization loss
        **kwargs
    ):
        super().__init__(num_classes=num_classes, init_lr=init_lr, **kwargs)
        self.save_hyperparameters()

        self.pretrained = pretrained
        self.input_channels = input_channels
        self._freeze_backbone_flag = freeze_backbone and pretrained
        self.use_se = use_se
        self.use_attention_pooling = use_attention_pooling
        self.use_localization_reg = use_localization_reg
        self.localization_reg_weight = localization_reg_weight

        # Stem mirrors the video ResNet stem: preserve depth resolution early
        self.conv1 = nn.Conv3d(input_channels, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        self.inplanes = 64
        self.layer1 = self._make_layer(64, blocks=2, stride=(1, 1, 1))
        # OPTIMIZATION: Preserve more depth information by only downsampling spatial dims in layer2
        self.layer2 = self._make_layer(128, blocks=2, stride=(1, 2, 2))  # Keep depth, downsample spatial
        self.layer3 = self._make_layer(256, blocks=2, stride=(2, 2, 2))  # Now downsample all dims
        self.layer4 = self._make_layer(512, blocks=2, stride=(2, 2, 2))

        # Pooling layer: either attention-based or standard average pooling
        if use_attention_pooling:
            self.pooling = SpatialAttentionPooling3D(512 * BasicBlock3D.expansion)
        else:
            self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
            self.pooling = None

        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512 * BasicBlock3D.expansion, num_classes)
        )

        if pretrained:
            self._inflate_from_2d()
        if self._freeze_backbone_flag:
            self._set_backbone_requires_grad(False)

    def _make_layer(self, planes: int, blocks: int, stride: tuple[int, int, int]) -> nn.Sequential:
        downsample = None
        stride = BasicBlock3D._to_3tuple(stride)

        if stride != (1, 1, 1) or self.inplanes != planes * BasicBlock3D.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * BasicBlock3D.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * BasicBlock3D.expansion)
            )

        layers = [BasicBlock3D(self.inplanes, planes, stride=stride, downsample=downsample, use_se=self.use_se)]
        self.inplanes = planes * BasicBlock3D.expansion
        for _ in range(1, blocks):
            layers.append(BasicBlock3D(self.inplanes, planes, use_se=self.use_se))
        return nn.Sequential(*layers)

    def _inflate_from_2d(self) -> None:
        """
        Inflate 2D ImageNet weights into the 3D backbone by repeating along the depth dimension.

        FIXED: Proper weight inflation without incorrect normalization that was breaking initialization.
        The key insight: we want each 3D filter slice to equal the 2D filter, so when applied
        to a volume, early layers behave similarly to applying 2D convs per-slice.
        """
        try:
            weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1  # type: ignore[attr-defined]
            resnet2d = torchvision.models.resnet18(weights=weights)
        except AttributeError:
            resnet2d = torchvision.models.resnet18(pretrained=True)

        state_2d = resnet2d.state_dict()
        state_3d = self.state_dict()

        for name, param2d in state_2d.items():
            if name not in state_3d:
                continue

            param3d = state_3d[name]
            if param2d.ndim == 4 and param3d.ndim == 5:
                depth = param3d.shape[2]
                # FIX: Don't divide by depth - this was breaking the initialization!
                # Each depth slice should have the full 2D filter weight
                inflated = param2d.unsqueeze(2).repeat(1, 1, depth, 1, 1)

                # Normalize by depth to make center slice have higher weight (optional alternative)
                # This makes the center of the temporal receptive field most important
                # weights = torch.zeros(depth)
                # weights[depth // 2] = 1.0  # Center slice gets full weight
                # inflated = inflated * weights.view(1, 1, depth, 1, 1)

                if inflated.shape[1] != param3d.shape[1]:
                    if param3d.shape[1] == 1:
                        # For single channel input (grayscale CT), average RGB channels
                        inflated = inflated.mean(dim=1, keepdim=True)
                    else:
                        repeat_factor = (param3d.shape[1] + inflated.shape[1] - 1) // inflated.shape[1]
                        inflated = inflated.repeat(1, repeat_factor, 1, 1, 1)
                        inflated = inflated[:, :param3d.shape[1], ...]

                state_3d[name] = inflated.to(param3d.dtype)
            elif param2d.shape == param3d.shape:
                state_3d[name] = param2d.to(param3d.dtype)

        missing, unexpected = self.load_state_dict(state_3d, strict=False)
        if missing:
            print(f"ResNet18_3D: skipped loading parameters {missing}")
        if unexpected:
            print(f"ResNet18_3D: unexpected keys when loading weights {unexpected}")

    def _set_backbone_requires_grad(self, requires_grad: bool) -> None:
        backbone_modules = [self.conv1, self.bn1, self.layer1, self.layer2, self.layer3, self.layer4]
        for module in backbone_modules:
            for param in module.parameters():
                param.requires_grad = requires_grad

    def unfreeze_backbone(self) -> None:
        self._set_backbone_requires_grad(True)

    def compute_localization_loss(self, attention_map: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Compute localization regularization loss to encourage attention to align with ground truth masks.

        Args:
            attention_map: Predicted spatial attention weights (B, 1, D', H', W')
            mask: Ground truth localization masks (B, 1, D, H, W)

        Returns:
            loss: Scalar localization regularization loss
        """
        # Resize mask to match attention map spatial dimensions
        if attention_map.shape != mask.shape:
            mask = F.interpolate(mask, size=attention_map.shape[2:], mode='trilinear', align_corners=False)

        # Normalize both to probability distributions with stronger epsilon for numerical stability
        eps = 1e-7
        attention_map_norm = attention_map / (attention_map.sum(dim=(2, 3, 4), keepdim=True) + eps)
        mask_norm = mask / (mask.sum(dim=(2, 3, 4), keepdim=True) + eps)

        # Clamp normalized values to prevent log(0) = -inf
        # This is CRITICAL: without clamping, log(0) produces -inf which becomes NaN
        attention_map_norm = torch.clamp(attention_map_norm, min=eps)
        mask_norm = torch.clamp(mask_norm, min=eps)

        # KL divergence loss: encourages attention to match ground truth localization
        # Only compute for samples that have localization annotations (non-zero masks)
        has_annotation = mask.sum(dim=(2, 3, 4)) > 0  # (B, 1)

        if has_annotation.sum() == 0:
            # No samples with annotations in this batch
            return torch.tensor(0.0, device=attention_map.device)

        # KL(mask || attention): penalizes attention not covering annotated regions
        # Now numerically stable: log is applied to clamped values (min=eps, never zero)
        kl_loss = F.kl_div(
            torch.log(attention_map_norm[has_annotation.squeeze()]),
            mask_norm[has_annotation.squeeze()],
            reduction='batchmean'
        )

        # Clamp the loss itself to prevent extreme values early in training
        kl_loss = torch.clamp(kl_loss, max=10.0)

        return kl_loss

    def training_step(self, batch, batch_idx):
        """
        Training step with optional localization regularization.
        """
        x, y = self.get_xy(batch)

        # Forward pass
        if self.use_localization_reg and 'mask' in batch:
            # Get attention map for localization regularization
            y_hat, features, attention_map = self.forward(x, return_attention=True)

            # Compute classification loss
            class_loss = self.loss(y_hat, y)

            # Compute localization regularization loss
            if attention_map is not None:
                loc_loss = self.compute_localization_loss(attention_map, batch['mask'])
                total_loss = class_loss + self.localization_reg_weight * loc_loss

                # Log both losses
                self.log('train_loss_class', class_loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
                self.log('train_loss_loc', loc_loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
            else:
                total_loss = class_loss
        else:
            # Standard forward pass without localization
            y_hat = self.forward(x)
            total_loss = self.loss(y_hat, y)

        # Compute and log metrics
        self.log('train_loss', total_loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log('train_acc', self.accuracy(y_hat, y), on_step=True, on_epoch=False, prog_bar=True)

        # Store outputs for epoch-level metrics
        self.training_outputs.append({
            'y_hat': y_hat.detach(),
            'y': y.detach()
        })

        return total_loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step with optional localization regularization.
        """
        x, y = self.get_xy(batch)

        # Forward pass
        if self.use_localization_reg and 'mask' in batch:
            y_hat, features, attention_map = self.forward(x, return_attention=True)

            # Compute classification loss
            class_loss = self.loss(y_hat, y)

            # Compute localization regularization loss
            if attention_map is not None:
                loc_loss = self.compute_localization_loss(attention_map, batch['mask'])
                total_loss = class_loss + self.localization_reg_weight * loc_loss

                # Log both losses
                self.log('val_loss_class', class_loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
                self.log('val_loss_loc', loc_loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
            else:
                total_loss = class_loss
        else:
            # Standard forward pass
            y_hat = self.forward(x)
            total_loss = self.loss(y_hat, y)

        # Compute and log metrics
        self.log('val_loss', total_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_acc', self.accuracy(y_hat, y), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        # Store outputs for epoch-level metrics
        self.validation_outputs.append({
            'y_hat': y_hat.detach(),
            'y': y.detach()
        })

        return total_loss

    def forward(self, x: torch.Tensor, return_attention: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Volumetric input tensor of shape (batch, channels, depth, height, width)
            return_attention: Whether to return intermediate features and attention map

        Returns:
            logits: Class logits of shape (batch, num_classes)
            features (optional): Feature map before pooling (batch, 512, D', H', W')
            attention_map (optional): Spatial attention weights (batch, 1, D', H', W') if using attention pooling
        """
        # OPTIMIZATION: Add normalization when using pretrained weights
        # The dataset.py already normalizes to mean=128.1722, std=87.1849
        # But ImageNet weights expect different normalization
        # Since we're using grayscale CT averaged from RGB weights, use grayscale ImageNet stats
        if self.pretrained and self.input_channels == 1:
            # Convert from dataset normalization (mean=128, std=87) back to [0,255] range
            # then apply ImageNet-style normalization
            # ImageNet grayscale equivalent: mean=0.449*255=114.5, std=0.226*255=57.6
            # Note: The NLST normalize already applied, so we work with that
            # This is approximate - ideally would renormalize, but this adds overhead
            pass  # Keep dataset normalization for now - the inflated weights should adapt

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        # Store features before pooling for potential localization regularization
        features = out

        # Apply either attention pooling or standard average pooling
        attention_map = None
        if self.pooling is not None:  # Attention pooling
            if return_attention or self.use_localization_reg:
                out, attention_map = self.pooling(out, return_attention=True)
            else:
                out = self.pooling(out)
        else:  # Standard average pooling
            out = self.avgpool(out)
            out = torch.flatten(out, 1)

        logits = self.fc(out)

        if return_attention:
            return logits, features, attention_map
        return logits


class ResNet18Video3D(Classifer):
    """
    Wrapper around torchvision's video ResNet-18 (r3d_18) for volumetric CT classification.

    Supports training from scratch or initializing from Kinetics-400 pretraining.
    """

    def __init__(
        self,
        input_channels: int = 1,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        num_classes: int = 2,
        init_lr: float = 1e-4,
        use_attention_pooling: bool = False,  # Use spatial attention pooling instead of avg pooling
        use_localization_reg: bool = False,  # Use localization masks as regularization
        localization_reg_weight: float = 0.1,  # Weight for localization regularization loss
        **kwargs
    ):
        super().__init__(num_classes=num_classes, init_lr=init_lr, **kwargs)
        self.save_hyperparameters()

        self.pretrained = pretrained
        self.input_channels = input_channels
        self._freeze_backbone_flag = freeze_backbone and pretrained
        self.use_attention_pooling = use_attention_pooling
        self.use_localization_reg = use_localization_reg
        self.localization_reg_weight = localization_reg_weight

        try:
            weights = torchvision.models.video.R3D_18_Weights.KINETICS400_V1  # type: ignore[attr-defined]
            self.backbone = torchvision.models.video.r3d_18(weights=weights if pretrained else None)
        except AttributeError:
            # Older torchvision API
            self.backbone = torchvision.models.video.r3d_18(pretrained=pretrained)

        # Replace first conv to accept arbitrary input channels while preserving pretrained weights when possible
        self._adapt_stem_conv(input_channels)

        # Get the number of features before the final FC layer
        in_features = self.backbone.fc.in_features

        # Replace the avgpool with optional attention pooling
        if use_attention_pooling:
            # Remove the original avgpool and replace with attention pooling
            self.backbone.avgpool = nn.Identity()  # Make avgpool a no-op
            self.attention_pooling = SpatialAttentionPooling3D(in_features)
        else:
            self.attention_pooling = None

        # Replace classification head
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, num_classes)
        )

        if self._freeze_backbone_flag:
            self.set_backbone_trainable(False)

    def _adapt_stem_conv(self, input_channels: int) -> None:
        orig_conv: nn.Conv3d = self.backbone.stem[0]
        if orig_conv.in_channels == input_channels:
            return

        new_conv = nn.Conv3d(
            input_channels,
            orig_conv.out_channels,
            kernel_size=orig_conv.kernel_size,
            stride=orig_conv.stride,
            padding=orig_conv.padding,
            bias=False
        )

        if self.pretrained:
            with torch.no_grad():
                weight = orig_conv.weight
                if input_channels == 1:
                    weight = weight.mean(dim=1, keepdim=True)
                elif input_channels > 3:
                    repeats = (input_channels + 2) // 3
                    weight = weight.repeat(1, repeats, 1, 1, 1)[:, :input_channels, ...]
                else:
                    weight = weight[:, :input_channels, ...]
                new_conv.weight.copy_(weight)

        self.backbone.stem[0] = new_conv

    def set_backbone_trainable(self, trainable: bool) -> None:
        for name, param in self.backbone.named_parameters():
            if "fc" in name:
                continue
            param.requires_grad = trainable

    def unfreeze_backbone(self) -> None:
        self.set_backbone_trainable(True)

    def compute_localization_loss(self, attention_map: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Compute localization regularization loss to encourage attention to align with ground truth masks.

        Args:
            attention_map: Predicted spatial attention weights (B, 1, D', H', W')
            mask: Ground truth localization masks (B, 1, D, H, W)

        Returns:
            loss: Scalar localization regularization loss
        """
        # Resize mask to match attention map spatial dimensions
        if attention_map.shape != mask.shape:
            mask = F.interpolate(mask, size=attention_map.shape[2:], mode='trilinear', align_corners=False)

        # Normalize both to probability distributions with stronger epsilon for numerical stability
        eps = 1e-7
        attention_map_norm = attention_map / (attention_map.sum(dim=(2, 3, 4), keepdim=True) + eps)
        mask_norm = mask / (mask.sum(dim=(2, 3, 4), keepdim=True) + eps)

        # Clamp normalized values to prevent log(0) = -inf
        # This is CRITICAL: without clamping, log(0) produces -inf which becomes NaN
        attention_map_norm = torch.clamp(attention_map_norm, min=eps)
        mask_norm = torch.clamp(mask_norm, min=eps)

        # KL divergence loss: encourages attention to match ground truth localization
        # Only compute for samples that have localization annotations (non-zero masks)
        has_annotation = mask.sum(dim=(2, 3, 4)) > 0  # (B, 1)

        if has_annotation.sum() == 0:
            # No samples with annotations in this batch
            return torch.tensor(0.0, device=attention_map.device)

        # KL(mask || attention): penalizes attention not covering annotated regions
        # Now numerically stable: log is applied to clamped values (min=eps, never zero)
        kl_loss = F.kl_div(
            torch.log(attention_map_norm[has_annotation.squeeze()]),
            mask_norm[has_annotation.squeeze()],
            reduction='batchmean'
        )

        # Clamp the loss itself to prevent extreme values early in training
        kl_loss = torch.clamp(kl_loss, max=10.0)

        return kl_loss

    def training_step(self, batch, batch_idx):
        """
        Training step with optional localization regularization.
        """
        x, y = self.get_xy(batch)

        # Forward pass
        if self.use_localization_reg and 'mask' in batch:
            # Get attention map for localization regularization
            y_hat, features, attention_map = self.forward(x, return_attention=True)

            # Compute classification loss
            class_loss = self.loss(y_hat, y)

            # Compute localization regularization loss
            if attention_map is not None:
                loc_loss = self.compute_localization_loss(attention_map, batch['mask'])
                total_loss = class_loss + self.localization_reg_weight * loc_loss

                # Log both losses
                self.log('train_loss_class', class_loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
                self.log('train_loss_loc', loc_loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
            else:
                total_loss = class_loss
        else:
            # Standard forward pass without localization
            y_hat = self.forward(x)
            total_loss = self.loss(y_hat, y)

        # Compute and log metrics
        self.log('train_loss', total_loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log('train_acc', self.accuracy(y_hat, y), on_step=True, on_epoch=False, prog_bar=True)

        # Store outputs for epoch-level metrics
        self.training_outputs.append({
            'y_hat': y_hat.detach(),
            'y': y.detach()
        })

        return total_loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step with optional localization regularization.
        """
        x, y = self.get_xy(batch)

        # Forward pass
        if self.use_localization_reg and 'mask' in batch:
            y_hat, features, attention_map = self.forward(x, return_attention=True)

            # Compute classification loss
            class_loss = self.loss(y_hat, y)

            # Compute localization regularization loss
            if attention_map is not None:
                loc_loss = self.compute_localization_loss(attention_map, batch['mask'])
                total_loss = class_loss + self.localization_reg_weight * loc_loss

                # Log both losses
                self.log('val_loss_class', class_loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
                self.log('val_loss_loc', loc_loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
            else:
                total_loss = class_loss
        else:
            # Standard forward pass
            y_hat = self.forward(x)
            total_loss = self.loss(y_hat, y)

        # Compute and log metrics
        self.log('val_loss', total_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_acc', self.accuracy(y_hat, y), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        # Store outputs for epoch-level metrics
        self.validation_outputs.append({
            'y_hat': y_hat.detach(),
            'y': y.detach()
        })

        return total_loss

    def forward(self, x: torch.Tensor, return_attention: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Volumetric input tensor of shape (batch, channels, depth, height, width)
            return_attention: Whether to return intermediate features and attention map

        Returns:
            logits: Class logits of shape (batch, num_classes)
            features (optional): Feature map before pooling (batch, 512, D', H', W')
            attention_map (optional): Spatial attention weights (batch, 1, D', H', W') if using attention pooling
        """
        # Extract features through the backbone (stem + residual blocks)
        x = self.backbone.stem(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        # Store features before pooling
        features = x

        # Apply pooling
        attention_map = None
        if self.attention_pooling is not None:
            # Use attention pooling
            if return_attention or self.use_localization_reg:
                x, attention_map = self.attention_pooling(x, return_attention=True)
            else:
                x = self.attention_pooling(x)
        else:
            # Use standard average pooling from backbone
            x = self.backbone.avgpool(x)
            x = torch.flatten(x, 1)

        # Classification head
        logits = self.backbone.fc(x)

        if return_attention:
            return logits, features, attention_map
        return logits


NLST_CENSORING_DIST = {
    "0": 0.9851928130104401,
    "1": 0.9748317321074379,
    "2": 0.9659923988537479,
    "3": 0.9587252204657843,
    "4": 0.9523590830936284,
    "5": 0.9461840310101468,
}

class CNN3D(Classifer):
    """
    3D Convolutional Neural Network (3D CNN) classifier for volumetric image classification.
    
    3D CNNs extend 2D CNNs to process volumetric data (like CT scans) by using 3D convolutions
    that operate across depth, height, and width dimensions simultaneously. This allows the model
    to learn spatial-temporal or 3D spatial patterns in medical imaging data.
    
    For NLST CT scans: input shape is (B, 1, D, H, W) where D=200 slices, H=W=256
    """
    
    def __init__(self, input_channels=1, hidden_dim=128, num_layers=2, num_classes=2, use_bn=True, init_lr=1e-3,
                 use_attention_pooling=False, use_localization_reg=False, localization_reg_weight=0.1, **kwargs):
        """
        Initialize the 3D CNN model.

        Args:
            input_channels (int): Number of input channels (1 for grayscale medical images)
            hidden_dim (int): Number of channels in convolutional layers
            num_layers (int): Number of convolutional blocks
            num_classes (int): Number of output classes for classification
            use_bn (bool): Whether to use batch normalization
            init_lr (float): Initial learning rate
            use_attention_pooling (bool): Whether to use spatial attention pooling instead of global average pooling
            use_localization_reg (bool): Whether to use localization regularization loss
            localization_reg_weight (float): Weight for localization regularization loss
        """
        super().__init__(num_classes=num_classes, init_lr=init_lr)
        self.save_hyperparameters()

        self.hidden_dim = hidden_dim
        self.use_bn = use_bn
        self.num_layers = num_layers
        self.use_attention_pooling = use_attention_pooling
        self.use_localization_reg = use_localization_reg
        self.localization_reg_weight = localization_reg_weight

        # Build 3D convolutional feature extractor
        conv_layers = []
        in_channels = input_channels

        for i in range(num_layers):
            # 3D Convolutional layer: applies learnable 3D filters to detect volumetric features
            # kernel_size=3 with padding=1 preserves spatial dimensions (after accounting for stride)
            conv_layers.append(nn.Conv3d(in_channels, hidden_dim, kernel_size=3, padding=1))

            # Optional 3D batch normalization for training stability
            if use_bn:
                conv_layers.append(nn.BatchNorm3d(hidden_dim))

            # ReLU activation for non-linearity
            conv_layers.append(nn.ReLU(inplace=True))

            # 3D Max pooling: reduces spatial dimensions (D, H, W) while keeping important features
            # Pool size 2x2x2 reduces each dimension by half
            conv_layers.append(nn.MaxPool3d(2, 2))

            # Update input channels for next layer
            in_channels = hidden_dim

            # Increase number of channels in deeper layers (common CNN pattern)
            if i < num_layers - 1:
                hidden_dim = min(hidden_dim * 2, 512)  # Cap at 512 channels

        self.conv_layers = nn.Sequential(*conv_layers)

        # Pooling layer: either spatial attention or global average pooling
        if self.use_attention_pooling:
            self.attention_pool = SpatialAttentionPooling3D(in_channels)
            self.global_pool = None
        else:
            self.global_pool = nn.AdaptiveAvgPool3d(1)
            self.attention_pool = None

        # Classifier head: fully-connected layer for final classification
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),  # Dropout for regularization
            nn.Linear(in_channels, num_classes)
        )

    def forward(self, x, return_attention=False):
        """
        Forward pass through the 3D CNN.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, depth, height, width)
                For NLST: (B, 1, 200, 256, 256)
            return_attention (bool): If True, return attention map along with logits (for localization reg)

        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes)
            OR
            tuple: (logits, features, attention_map) if return_attention=True
        """
        # Extract features using 3D convolutional layers
        # Each conv block: Conv3d -> BatchNorm3d -> ReLU -> MaxPool3d
        # Input: (B, C, D, H, W) -> features: (B, hidden_dim, D', H', W') where dimensions are reduced
        features = self.conv_layers(x)

        # Pooling: either attention-based or global average pooling
        attention_map = None
        if self.use_attention_pooling:
            # Spatial attention pooling with learnable attention weights
            if return_attention and self.use_localization_reg:
                pooled, attention_map = self.attention_pool(features, return_attention=True)
            else:
                pooled = self.attention_pool(features, return_attention=False)
        else:
            # Global average pooling: (B, C, D', H', W') -> (B, C, 1, 1, 1)
            pooled = self.global_pool(features)
            pooled = pooled.view(pooled.size(0), -1)

        # Flatten for fully-connected layer if needed: (B, C, 1, 1, 1) -> (B, C)
        if len(pooled.shape) > 2:
            flattened = pooled.view(pooled.size(0), -1)
        else:
            flattened = pooled

        # Final classification: (B, C) -> (B, num_classes)
        logits = self.classifier(flattened)

        if return_attention:
            return logits, features, attention_map
        return logits

    def compute_localization_loss(self, attention_map: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Compute localization regularization loss to encourage attention to align with ground truth masks.

        Args:
            attention_map: Predicted spatial attention weights (B, 1, D', H', W')
            mask: Ground truth localization masks (B, 1, D, H, W)

        Returns:
            loss: Scalar localization regularization loss
        """
        # Resize mask to match attention map spatial dimensions
        if attention_map.shape != mask.shape:
            mask = F.interpolate(mask, size=attention_map.shape[2:], mode='trilinear', align_corners=False)

        # Normalize both to probability distributions with stronger epsilon for numerical stability
        eps = 1e-7
        attention_map_norm = attention_map / (attention_map.sum(dim=(2, 3, 4), keepdim=True) + eps)
        mask_norm = mask / (mask.sum(dim=(2, 3, 4), keepdim=True) + eps)

        # Clamp normalized values to prevent log(0) = -inf
        # This is CRITICAL: without clamping, log(0) produces -inf which becomes NaN
        attention_map_norm = torch.clamp(attention_map_norm, min=eps)
        mask_norm = torch.clamp(mask_norm, min=eps)

        # KL divergence loss: encourages attention to match ground truth localization
        # Only compute for samples that have localization annotations (non-zero masks)
        has_annotation = mask.sum(dim=(2, 3, 4)) > 0  # (B, 1)

        if has_annotation.sum() == 0:
            # No samples with annotations in this batch
            return torch.tensor(0.0, device=attention_map.device)

        # KL(mask || attention): penalizes attention not covering annotated regions
        # Now numerically stable: log is applied to clamped values (min=eps, never zero)
        kl_loss = F.kl_div(
            torch.log(attention_map_norm[has_annotation.squeeze()]),
            mask_norm[has_annotation.squeeze()],
            reduction='batchmean'
        )

        # Clamp the loss itself to prevent extreme values early in training
        kl_loss = torch.clamp(kl_loss, max=10.0)

        return kl_loss

    def training_step(self, batch, batch_idx):
        """
        Training step with optional localization regularization.
        """
        x, y = self.get_xy(batch)

        # Forward pass
        if self.use_localization_reg and 'mask' in batch:
            # Get attention map for localization regularization
            y_hat, features, attention_map = self.forward(x, return_attention=True)

            # Compute classification loss
            class_loss = self.loss(y_hat, y)

            # Compute localization regularization loss
            if attention_map is not None:
                loc_loss = self.compute_localization_loss(attention_map, batch['mask'])
                total_loss = class_loss + self.localization_reg_weight * loc_loss

                # Log both losses
                self.log('train_loss_class', class_loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
                self.log('train_loss_loc', loc_loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
            else:
                total_loss = class_loss
        else:
            # Standard forward pass without localization
            y_hat = self.forward(x)
            total_loss = self.loss(y_hat, y)

        # Compute and log metrics
        self.log('train_loss', total_loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log('train_acc', self.accuracy(y_hat, y), on_step=True, on_epoch=False, prog_bar=True)

        # Store outputs for epoch-level metrics
        self.training_outputs.append({
            'y_hat': y_hat.detach(),
            'y': y.detach()
        })

        return total_loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step with optional localization regularization.
        """
        x, y = self.get_xy(batch)

        # Forward pass
        attention_map = None
        if (self.use_localization_reg or self.use_attention_pooling) and 'mask' in batch:
            y_hat, features, attention_map = self.forward(x, return_attention=True)

            # Compute classification loss
            class_loss = self.loss(y_hat, y)

            # Compute localization regularization loss
            if attention_map is not None and self.use_localization_reg:
                loc_loss = self.compute_localization_loss(attention_map, batch['mask'])
                total_loss = class_loss + self.localization_reg_weight * loc_loss

                # Log both losses
                self.log('val_loss_class', class_loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
                self.log('val_loss_loc', loc_loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
            else:
                total_loss = class_loss
        else:
            # Standard forward pass without localization
            y_hat = self.forward(x)
            total_loss = self.loss(y_hat, y)

        # Compute and log metrics
        self.log('val_loss', total_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_acc', self.accuracy(y_hat, y), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        # Store outputs for epoch-level metrics and visualization
        output_dict = {
            'y_hat': y_hat.detach(),
            'y': y.detach()
        }

        # Store attention maps and masks for visualization (only on rank 0, and limit to 4 samples)
        # This prevents memory explosion and DDP sync issues
        if attention_map is not None and 'mask' in batch and self.trainer.is_global_zero:
            # Only store if we haven't collected enough samples yet
            num_viz_samples = sum(1 for o in self.validation_outputs if 'attention_map' in o)
            if num_viz_samples < 4:
                output_dict['attention_map'] = attention_map.detach()
                output_dict['mask'] = batch['mask'].detach()
                output_dict['ct_volume'] = x.detach()

        self.validation_outputs.append(output_dict)

        return total_loss

    def on_validation_epoch_end(self):
        """
        Called at the end of validation epoch. Compute AUC and create localiz ation visualizations.
        """
        # Concatenate all predictions and labels from the epoch
        y_hat = torch.cat([o["y_hat"] for o in self.validation_outputs])
        y = torch.cat([o["y"] for o in self.validation_outputs])

        # Convert logits to probabilities
        if self.num_classes == 2:
            probs = F.softmax(y_hat, dim=-1)[:, self.positive_class]
        else:
            probs = F.softmax(y_hat, dim=-1)

        # Log validation AUC
        self._log_epoch_auc("val", probs, y.view(-1))

        # Create localization visualizations and compute metrics (only on rank 0 to avoid DDP hangs)
        if self.use_attention_pooling and len(self.validation_outputs) > 0 and 'attention_map' in self.validation_outputs[0]:
            # Only create visualizations on the main process (rank 0) to avoid distributed training hangs
            if self.trainer.is_global_zero:
                from src.localization_utils import log_localization_visualizations, compute_localization_metrics

                # Gather samples with attention maps (limit to avoid memory issues)
                samples_with_attention = [o for o in self.validation_outputs if 'attention_map' in o][:4]

                if len(samples_with_attention) > 0:
                    ct_volumes = torch.cat([o['ct_volume'] for o in samples_with_attention])
                    attention_maps = torch.cat([o['attention_map'] for o in samples_with_attention])
                    masks = torch.cat([o['mask'] for o in samples_with_attention])

                    # Compute localization metrics
                    loc_metrics = compute_localization_metrics(attention_maps, masks)
                    if loc_metrics['num_samples'] > 0:
                        self.log('val_iou', loc_metrics['iou'], prog_bar=True, sync_dist=False)
                        self.log('val_dice', loc_metrics['dice'], prog_bar=True, sync_dist=False)

                    # Create visualizations
                    log_localization_visualizations(
                        logger=self.logger,
                        ct_volumes=ct_volumes,
                        attention_maps=attention_maps,
                        masks=masks,
                        stage='val',
                        epoch=self.current_epoch,
                        max_samples=4
                    )

        # Clear stored outputs
        self.validation_outputs = []

class RiskModel(Classifer):
    def __init__(self, input_num_chan=1, num_classes=2, init_lr = 1e-3, max_followup=6, **kwargs):
        super().__init__(num_classes=num_classes, init_lr=init_lr)
        self.save_hyperparameters()

        self.hidden_dim = 512

        ## Maximum number of followups to predict (set to 6 for full risk prediction task)
        self.max_followup = max_followup

        # TODO: Initalize components of your model here
        raise NotImplementedError("Not implemented yet")



    def forward(self, x):
        raise NotImplementedError("Not implemented yet")

    def get_xy(self, batch):
        """
            x: (B, C, D, W, H) -  Tensor of CT volume
            y_seq: (B, T) - Tensor of cancer outcomes. a vector of [0,0,1,1,1, 1] means the patient got between years 2-3, so
            had cancer within 3 years, within 4, within 5, and within 6 years.
            y_mask: (B, T) - Tensor of mask indicating future time points are observed and not censored. For example, if y_seq = [0,0,0,0,0,0], then y_mask = [1,1,0,0,0,0], we only know that the patient did not have cancer within 2 years, but we don't know if they had cancer within 3 years or not.
            mask: (B, D, W, H) - Tensor of mask indicating which voxels are inside an annotated cancer region (1) or not (0).
                TODO: You can add more inputs here if you want to use them from the NLST dataloader.
                Hint: You may want to change the mask definition to suit your localization method

        """
        return batch['x'], batch['y_seq'][:, :self.max_followup], batch['y_mask'][:, :self.max_followup], batch['mask']

    def step(self, batch, batch_idx, stage, outputs):
        x, y_seq, y_mask, region_annotation_mask = self.get_xy(batch)

        # TODO: Get risk scores from your model
        y_hat = None ## (B, T) shape tensor of risk scores.
        # TODO: Compute your loss (with or without localization)
        loss = None

        raise NotImplementedError("Not implemented yet")
        
        # TODO: Log any metrics you want to wandb
        metric_value = -1
        metric_name = "dummy_metric"
        self.log('{}_{}'.format(stage, metric_name), metric_value, prog_bar=True, on_epoch=True, on_step=True, sync_dist=True)

        # TODO: Store the predictions and labels for use at the end of the epoch for AUC and C-Index computation.
        outputs.append({
            "y_hat": y_hat, # Logits for all risk scores
            "y_mask": y_mask, # Tensor of when the patient was observed
            "y_seq": y_seq, # Tensor of when the patient had cancer
            "y": batch["y"], # If patient has cancer within 6 years
            "time_at_event": batch["time_at_event"] # Censor time
        })

        return loss
    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "train", self.training_outputs)

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "val", self.validation_outputs)
    def test_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "test", self.test_outputs)

    def on_epoch_end(self, stage, outputs):
        y_hat = F.sigmoid(torch.cat([o["y_hat"] for o in outputs]))
        y_seq = torch.cat([o["y_seq"] for o in outputs])
        y_mask = torch.cat([o["y_mask"] for o in outputs])

        for i in range(self.max_followup):
            '''
                Filter samples for either valid negative (observed followup) at time i
                or known pos within range i (including if cancer at prev time and censoring before current time)
            '''
            valid_probs = y_hat[:, i][(y_mask[:, i] == 1) | (y_seq[:,i] == 1)]
            valid_labels = y_seq[:, i][(y_mask[:, i] == 1)| (y_seq[:,i] == 1)]
            self.log("{}_{}year_auc".format(stage, i+1), self.auc(valid_probs, valid_labels.view(-1)), sync_dist=True, prog_bar=True)

        y = torch.cat([o["y"] for o in outputs])
        time_at_event = torch.cat([o["time_at_event"] for o in outputs])

        if y.sum() > 0 and self.max_followup == 6:
            c_index = concordance_index(time_at_event.cpu().numpy(), y_hat.detach().cpu().numpy(), y.cpu().numpy(), NLST_CENSORING_DIST)
        else:
            c_index = 0
        self.log("{}_c_index".format(stage), c_index, sync_dist=True, prog_bar=True)

    def on_train_epoch_end(self):
        self.on_epoch_end("train", self.training_outputs)
        self.training_outputs = []

    def on_validation_epoch_end(self):
        self.on_epoch_end("val", self.validation_outputs)
        self.validation_outputs = []

    def on_test_epoch_end(self):
        self.on_epoch_end("test", self.test_outputs)
        self.test_outputs = []
